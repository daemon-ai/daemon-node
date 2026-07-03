{
  description = "daemon Rust workspace";

  inputs = {
    nixpkgs.url = "github:logos-co/nixpkgs/mingw-integration";
    flake-utils.url = "github:numtide/flake-utils";
    crane.url = "github:ipetkov/crane";

    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    # Pinned to the exact llama.cpp commit vendored by `llama-cpp-sys-4` 0.3.2 (commit `94a220cd6`,
    # tag `b9496`, per that crate's README), so a from-source build here is ABI-compatible with the
    # crate's bindgen output. Bump in lockstep whenever the `llama-cpp-4` dependency is upgraded.
    llama-cpp-src = {
      url = "github:ggml-org/llama.cpp/94a220cd6745e6e3f8de62870b66fd5b9bc92700";
      flake = false;
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      crane,
      fenix,
      llama-cpp-src,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        lib = pkgs.lib;
        caBundle = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";

        # Version: the SemVer base lives in `./VERSION` (mirrored by `[workspace.package].version`
        # in Cargo.toml; the `just check-version` gate asserts they agree). The build-metadata
        # identifier is derived from the flake source revision (no `.git` in the sandbox), retaining
        # the off-tag / dirty marker like phosphor: `g<rev>` clean, `g<rev>.dirty` dirty, else a
        # narHash fallback. It is wrapped as `+<id>` by daemon-common's build.rs (via DAEMON_BUILD_ID).
        baseVersion = lib.strings.trim (builtins.readFile ./VERSION);
        buildId =
          if self ? shortRev then
            "g${self.shortRev}"
          else if self ? dirtyShortRev then
            "g${lib.removeSuffix "-dirty" self.dirtyShortRev}.dirty"
          else
            "nar${builtins.substring 0 8 (lib.removePrefix "sha256-" (self.narHash or "sha256-unknown"))}";

        # Native build inputs for the `daemon-infer` worker's optional engine backends
        # (llama-cpp-4 / mistral.rs). These are only consumed when a worker is built with an engine
        # feature (`--features llama`, etc.); the default workspace gate compiles only the stub worker
        # and never touches cmake/clang. `libclang` is required by bindgen (llama-cpp-sys-4).
        engineNativeInputs = [
          pkgs.cmake
          pkgs.clang
          pkgs.llvmPackages.libclang.lib
          pkgs.pkg-config
        ];
        libclangPath = "${pkgs.llvmPackages.libclang.lib}/lib";

        # A from-source, shared-lib build of the exact llama.cpp commit that `llama-cpp-sys-4` 0.3.2
        # vendors, with BOTH the CPU and Vulkan ggml backends compiled in (`GGML_VULKAN=ON`; the CPU
        # backend is always built). One artifact serves every dev lane: with `n_gpu_layers = 0` it
        # runs on CPU, and with layers offloaded it uses Vulkan if a device is present (else it falls
        # back to CPU). cmake runs in the Nix build sandbox (which has `/bin/sh`), so this sidesteps
        # the missing-`/bin/sh` trap. Pointing `LLAMA_PREBUILT_DIR` at it lets a plain `cargo build
        # --features llama,dynamic-link` skip cmake entirely in the dev shell (only the `cc`-built
        # `mtp_shim` compiles locally). `GGML_NATIVE=OFF` keeps the store path portable/reproducible;
        # `shaderc` provides `glslc` for the SPIR-V shader compile, and `vulkan-headers` +
        # `vulkan-loader` satisfy cmake's `FindVulkan` and the `libggml-vulkan.so` link. At runtime the
        # NixOS loader resolves the system ICD (RADV) from `/run/opengl-driver/share/vulkan/icd.d`.
        llamaCpp = pkgs.stdenv.mkDerivation {
          pname = "llama-cpp-prebuilt";
          version = "b9496";
          src = llama-cpp-src;
          nativeBuildInputs = [ pkgs.cmake pkgs.ninja pkgs.pkg-config pkgs.shaderc ];
          buildInputs = [ pkgs.vulkan-headers pkgs.vulkan-loader ];
          # Build only the ggml + libllama shared libraries; everything else (tools, the unified
          # `app` binary, server, examples, tests, common, web UI) is dropped — we only consume the
          # `.so`s, and those extra targets pull in deps that fail to link in this trimmed build.
          cmakeFlags = [
            "-DBUILD_SHARED_LIBS=ON"
            "-DGGML_NATIVE=OFF"
            "-DGGML_VULKAN=ON"
            "-DLLAMA_CURL=OFF"
            "-DLLAMA_BUILD_TESTS=OFF"
            "-DLLAMA_BUILD_EXAMPLES=OFF"
            "-DLLAMA_BUILD_SERVER=OFF"
            "-DLLAMA_BUILD_TOOLS=OFF"
            "-DLLAMA_BUILD_APP=OFF"
            "-DLLAMA_BUILD_COMMON=OFF"
            "-DLLAMA_BUILD_HTML=OFF"
            "-DLLAMA_BUILD_UI=OFF"
          ];
        };

        rustToolchain = fenix.packages.${system}.stable.withComponents [
          "cargo"
          "clippy"
          "rust-src"
          "rustc"
          "rustfmt"
          # llvm-tools (llvm-profdata/llvm-cov) backs `cargo llvm-cov` coverage runs.
          "llvm-tools"
        ];

        # Nightly toolchain for the lanes that require it: Miri (UB checking over the FFI/codec
        # `unsafe` surface) and cargo-fuzz (libFuzzer needs `-Z` flags). Kept off the default shell
        # so the everyday `cargo build/test` stays on the pinned stable toolchain; exposed via the
        # separate `nightly` devShell below.
        rustNightly = fenix.packages.${system}.complete.withComponents [
          "cargo"
          "clippy"
          "rustc"
          "rustfmt"
          "rust-src"
          "miri"
          "llvm-tools"
        ];

        craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;

        # crane's `cleanCargoSource` keeps only Rust/Cargo files, which drops the non-`.rs` grammar
        # assets `daemon-infer` embeds via `include_str!` (`metta.gbnf` / `metta.lark`) — a from-source
        # `nix build` then fails to compile that crate. Extend the Cargo-source filter to also retain
        # those extensions so the sandbox build sees them. Keep this in lockstep with any new
        # `include_str!`/`include_bytes!` of non-Rust assets in the workspace.
        src = lib.cleanSourceWith {
          src = ./.;
          name = "source";
          filter =
            path: type:
            (craneLib.filterCargoSources path type)
            # `daemon-infer` embeds .gbnf/.lark grammars via include_str!; the codec toolchain
            # (`xtask verify-codec` / the superproject codegen derivation) needs the CDDL, the
            # canonical codegen script, and the ciborium fixtures it decodes.
            || (builtins.match ".*\\.(gbnf|lark|cddl|cbor|sh)$" path != null);
        };

        # hyperon (MeTTa) is a *git* dependency (no crates.io release). crane's default vendoring would
        # re-fetch it with a `fetchgit` hash we don't have; instead (vendoring "Option A") we pin the
        # checkout to `fetchFromGitHub` at the exact rev in `Cargo.toml`, using the prefetched
        # `nix flake prefetch` hash. The repo provides several crates (hyperon, hyperon-atom,
        # hyperon-space, hyperon-common, hyperon-macros) from this one checkout, so the override keys
        # off any `hyperon*` package sharing the git source.
        hyperonSrc = pkgs.fetchFromGitHub {
          owner = "trueagi-io";
          repo = "hyperon-experimental";
          rev = "3f76dc460da6961f57f69f6c3e550c59c74ada83";
          hash = "sha256-qTx32OBwtcytMPbPTnhNUD+Eccir3oFQhpjPgyfa5IA=";
        };

        # The vendored Cargo registry+git sources, with the hyperon git checkout swapped for the
        # `fetchFromGitHub` source above. Shared by `buildDepsOnly` and every `buildPackage` via
        # `commonArgs.cargoVendorDir`, so the pin is consistent across the default gate and the
        # hyperon worker lane.
        cargoVendorDir = craneLib.vendorCargoDeps {
          inherit src;
          overrideVendorGitCheckout =
            ps: drv:
            if lib.any (p: lib.hasPrefix "hyperon" p.name) ps then
              drv.overrideAttrs (_: { src = hyperonSrc; })
            else
              drv;
        };

        commonArgs = {
          inherit src cargoVendorDir;
          pname = "daemon-workspace";
          version = baseVersion;
          strictDeps = true;
          SSL_CERT_FILE = caBundle;
          NIX_SSL_CERT_FILE = caBundle;
        };

        cargoArtifacts = craneLib.buildDepsOnly commonArgs;

        buildWorkspacePackage =
          packageName:
          craneLib.buildPackage (
            commonArgs
            // {
              pname = packageName;
              version = baseVersion;
              inherit cargoArtifacts;
              cargoExtraArgs = "-p ${packageName}";
              # Inject the reproducible build-metadata id here (not in commonArgs) so the shared
              # dependency artifacts stay cached across revisions; only the final crates rebuild
              # when the revision changes. daemon-common's build.rs reads this.
              DAEMON_BUILD_ID = buildId;
            }
          );

        daemon = buildWorkspacePackage "daemon";
        daemon-cli = buildWorkspacePackage "daemon-cli";

        # Engine-lane compile checks: build the `daemon-infer` worker with an engine feature so the
        # llama-cpp-4 / mistral.rs glue is type-checked against the real native APIs. These compile
        # C/C++ via cmake, which needs `/bin/sh` + a full stdenv — provided by the nix build sandbox
        # (a raw `cargo` build in the dev shell can't satisfy make's hardcoded `/bin/sh` on NixOS).
        # They are deliberately separate outputs, NOT part of the default workspace gate.
        #
        # The statically-compiled llama.cpp links OpenMP, whose runtime (libgomp) the Rust link
        # step does not record on the RUNPATH — patch it in so the packaged worker runs stand-alone
        # (the superproject bundle ships it next to the daemon; without this it dies on load with
        # "libgomp.so.1: cannot open shared object file").
        buildEngineWorker =
          name: features:
          craneLib.buildPackage (
            commonArgs
            // {
              pname = "daemon-infer-${name}";
              inherit cargoArtifacts;
              cargoExtraArgs = "-p daemon-infer --features ${features}";
              nativeBuildInputs = engineNativeInputs ++ [ pkgs.patchelf ];
              LIBCLANG_PATH = libclangPath;
              doCheck = false;
              postInstall = ''
                patchelf --add-rpath ${pkgs.gcc.cc.lib}/lib "$out/bin/daemon-infer"
              '';
            }
          );

        # The llama lane ships with multimodal projector loading (`mtmd`): this is a from-source
        # cmake build inside the sandbox, so tools/mtmd compiles in (unlike the dev-shell prebuilt,
        # which stays LLAMA_BUILD_TOOLS=OFF and has no libmtmd — keep `mtmd` out of dev-shell
        # `dynamic-link` builds).
        daemon-infer-llama = buildEngineWorker "llama" "llama,mtmd";
        daemon-infer-mistralrs = buildEngineWorker "mistralrs" "mistralrs";

        # The authoritative "llama-cpp-4 compiles with the Vulkan backend" gate: build the worker
        # `--features vulkan`, which forwards to `llama-cpp-4/vulkan` -> `llama-cpp-sys-4/vulkan` ->
        # `GGML_VULKAN=ON` in cmake. Unlike the CPU lanes this needs the Vulkan SDK pieces at build
        # time: `shaderc` (`glslc`) to compile the SPIR-V shaders, and `vulkan-headers` +
        # `vulkan-loader` for cmake's `FindVulkan` and the `libggml-vulkan` link. Compile-only
        # (`doCheck = false`); runtime GPU exercise happens via the dev-shell tests.
        daemon-infer-vulkan = craneLib.buildPackage (
          commonArgs
          // {
            pname = "daemon-infer-vulkan";
            inherit cargoArtifacts;
            cargoExtraArgs = "-p daemon-infer --features vulkan";
            nativeBuildInputs = engineNativeInputs ++ [ pkgs.shaderc ];
            # `spirv-headers` satisfies the `find_package(SPIRV-Headers)` in the crate's vendored
            # `ggml-vulkan` CMake (and its `license_add_file`); without it the from-source Vulkan
            # build fails at configure time.
            buildInputs = [ pkgs.vulkan-headers pkgs.vulkan-loader pkgs.spirv-headers ];
            LIBCLANG_PATH = libclangPath;
            doCheck = false;
          }
        );

        # The MeTTa symbolic-coprocessor worker, built WITH the real engine (`--features hyperon`).
        # This is a deliberately separate output, NOT part of the default workspace gate: the default
        # `daemon-metta` (fallback engine) and every other crate never link `hyperon`. The hyperon
        # build pulls only crates.io deps (the `pkg_mgmt` feature: serde/serde_json/semver/xxhash) —
        # no `git2`/libgit2 and no second git source (`das`/metta-bus-client are not enabled) — so no
        # extra native build inputs are required beyond the Rust toolchain. `pkg-config` is included
        # defensively for any transitive sys-crate probe.
        daemon-metta = craneLib.buildPackage (
          commonArgs
          // {
            pname = "daemon-metta";
            version = baseVersion;
            inherit cargoArtifacts;
            cargoExtraArgs = "-p daemon-metta --features hyperon";
            nativeBuildInputs = [ pkgs.pkg-config ];
            doCheck = false;
          }
        );
      in
      {
        packages = {
          inherit
            daemon
            daemon-cli
            daemon-infer-llama
            daemon-infer-mistralrs
            daemon-infer-vulkan
            daemon-metta
            ;
          # Prebuilt llama.cpp (shared, CPU + Vulkan) matching the crate's vendored commit; consumed
          # by the dev shells via `LLAMA_PREBUILT_DIR` to compile the llama lane without cmake.
          llama-cpp = llamaCpp;
          default = daemon;
        };

        apps = {
          daemon = (flake-utils.lib.mkApp {
            drv = daemon;
            name = "daemon";
          }) // {
            meta.description = "Run the daemon host binary";
          };
          daemon-cli = (flake-utils.lib.mkApp {
            drv = daemon-cli;
            name = "daemon-cli";
          }) // {
            meta.description = "Run the daemon operator CLI";
          };
          default = self.apps.${system}.daemon;
        };

        checks = {
          inherit daemon daemon-cli;
          # Prove the generated zcbor C codec round-trips real ciborium wire bytes (`xtask
          # verify-codec`): generate the codec, compile its decoder + the zcbor runtime, and decode
          # every fixture. This is the loop the syntactic `cddl` parity gate cannot close. Needs
          # zcbor (codegen) + cc (from stdenv) at build time.
          verify-codec = craneLib.mkCargoDerivation (
            commonArgs
            // {
              pname = "daemon-verify-codec";
              version = baseVersion;
              inherit cargoArtifacts;
              doInstallCargoArtifacts = false;
              buildPhaseCargoCommand = "cargo run -p xtask -- verify-codec";
              nativeBuildInputs = [ pkgs.python3Packages.zcbor ];
            }
          );
        };

        devShells = {
          default = craneLib.devShell {
            # Worker engine toolchain (clang/libclang for bindgen, cmake for the GPU lanes) is present
            # so a dev can build an engine lane locally. The default `cargo test --workspace` still
            # builds only the stub worker — no engine, no cmake step.
            #
            # llama lane in the dev shell: rather than compile llama.cpp from source here, we point
            # `LLAMA_PREBUILT_DIR` at the pinned `packages.llama-cpp` (built from source in the Nix
            # sandbox), so `cargo build -p daemon-infer --features llama,dynamic-link` links that
            # prebuilt and skips cmake entirely (only the `cc`-built `mtp_shim` compiles locally).
            # That prebuilt now bundles the Vulkan backend (one CPU+Vulkan artifact), so
            # `LD_LIBRARY_PATH` also includes the Vulkan loader: `libggml-vulkan.so` has a `DT_NEEDED`
            # on `libvulkan.so`, which must resolve even for a CPU-only (`n_gpu_layers = 0`) run. It
            # also makes the shared llama/ggml libs + libgomp resolvable when the worker runs in-shell
            # (e.g. `cargo test`, `cargo run`).
            LIBCLANG_PATH = libclangPath;
            LLAMA_PREBUILT_DIR = "${llamaCpp}";
            LLAMA_PREBUILT_SHARED = "1";
            LD_LIBRARY_PATH = "${llamaCpp}/lib:${pkgs.vulkan-loader}/lib:${pkgs.gcc.cc.lib}/lib";
            SSL_CERT_FILE = caBundle;
            NIX_SSL_CERT_FILE = caBundle;
            packages =
              [
                rustToolchain
                fenix.packages.${system}.rust-analyzer
                pkgs.rust-cbindgen
                pkgs.python3Packages.zcbor
                # --- code-quality tooling (see justfile `lint` / `audit-cleanup` / `coverage`) ---
                pkgs.cargo-deny # advisories + license/ban/source policy (supersedes cargo-audit)
                pkgs.cargo-nextest # faster, more reliable test runner
                pkgs.cargo-machete # fast unused-dependency detection
                pkgs.cargo-hack # feature-powerset checks across the many feature gates
                pkgs.cargo-mutants # mutation testing (validates test strength)
                pkgs.cargo-llvm-cov # source-based coverage
                pkgs.gitleaks # secret scanning
                pkgs.typos # source spell-checker
                pkgs.nodejs # provides npx for jscpd (not packaged in nixpkgs)
                pkgs.just # task runner: the justfile recipes (lint / deny / test / coverage)
              ]
              ++ engineNativeInputs;
          };
        }
        // {
          # Nightly lane for Miri + cargo-fuzz (`just miri` / `just fuzz`). Separate from the default
          # shell so the everyday build never pulls the nightly toolchain onto PATH.
          nightly = pkgs.mkShell {
            LIBCLANG_PATH = libclangPath;
            SSL_CERT_FILE = caBundle;
            NIX_SSL_CERT_FILE = caBundle;
            packages = [
              rustNightly
              pkgs.cargo-fuzz
            ] ++ engineNativeInputs;
          };
        }
        # Optional GPU lanes for building/exercising the worker with an accelerated backend
        # (`cargo build -p daemon-infer --features cuda` / `--features vulkan`). Linux-only; the CUDA
        # shell needs an unfree-permitting nixpkgs, so it is built lazily and only on Linux.
        // lib.optionalAttrs pkgs.stdenv.isLinux {
          vulkan = craneLib.devShell {
            LIBCLANG_PATH = libclangPath;
            # Same CPU+Vulkan prebuilt as the default shell, so `cargo build -p daemon-infer
            # --features llama,dynamic-link` links a Vulkan-capable llama.cpp and skips cmake.
            # `LD_LIBRARY_PATH` resolves the shared llama/ggml libs (incl. `libggml-vulkan.so`), the
            # Vulkan loader (`libvulkan.so` -> RADV ICD under `/run/opengl-driver`), and libgomp at
            # runtime. The Vulkan SDK pieces on `packages` also let a from-source `--features vulkan`
            # build work here (it just recompiles llama.cpp via the crate's cmake path).
            LLAMA_PREBUILT_DIR = "${llamaCpp}";
            LLAMA_PREBUILT_SHARED = "1";
            LD_LIBRARY_PATH = "${llamaCpp}/lib:${pkgs.vulkan-loader}/lib:${pkgs.gcc.cc.lib}/lib";
            SSL_CERT_FILE = caBundle;
            NIX_SSL_CERT_FILE = caBundle;
            packages =
              [ rustToolchain pkgs.rust-cbindgen ]
              ++ engineNativeInputs
              ++ [ pkgs.vulkan-headers pkgs.vulkan-loader pkgs.shaderc ];
          };
          cuda =
            let
              cudaPkgs = import nixpkgs {
                inherit system;
                config.allowUnfree = true;
              };
            in
            craneLib.devShell {
              LIBCLANG_PATH = libclangPath;
              CUDA_PATH = "${cudaPkgs.cudatoolkit}";
              SSL_CERT_FILE = caBundle;
              NIX_SSL_CERT_FILE = caBundle;
              packages =
                [ rustToolchain pkgs.rust-cbindgen ]
                ++ engineNativeInputs
                ++ [ cudaPkgs.cudatoolkit ];
            };
        };
      }
    );
}
