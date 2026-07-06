{
  description = "daemon Rust workspace";

  # Pull built closures from the daemon-ai Cachix cache (public pull). CI feeds the cache via
  # cachix-action; humans/other machines opt in with --accept-flake-config (or as a trusted-user).
  # Public pull key only — no secret lives here.
  nixConfig = {
    extra-substituters = [ "https://daemon-ai.cachix.org" ];
    extra-trusted-public-keys = [ "daemon-ai.cachix.org-1:jzeLmFDfgE5dzGT0RXF70IEU/tKsWdDV9LQ5zPGAnQs=" ];
  };

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
          # Build the ggml + libllama shared libraries AND libmtmd (the multimodal projector loader).
          # `mtmd` lives under tools/, so it only builds with LLAMA_BUILD_TOOLS=ON + the common lib
          # (LLAMA_BUILD_COMMON=ON) — mirroring the proven Windows lane below. Everything else (the
          # unified `app` binary, server, examples, tests, web UI) stays off; `MTMD_VIDEO=OFF` avoids
          # the optional ffmpeg dependency. TOOLS=ON also drags tool byproducts (`*-impl` libs,
          # libcommon) into the install; those are pruned in postInstall because `llama-cpp-sys-4`'s
          # prebuilt mode GLOBS every lib in `$out/lib` and links them all — the dir must hold exactly
          # the core runtime `.so`s (llama/ggml*/mtmd), nothing more.
          cmakeFlags = [
            "-DBUILD_SHARED_LIBS=ON"
            "-DGGML_NATIVE=OFF"
            "-DGGML_VULKAN=ON"
            "-DLLAMA_CURL=OFF"
            "-DLLAMA_BUILD_TESTS=OFF"
            "-DLLAMA_BUILD_EXAMPLES=OFF"
            "-DLLAMA_BUILD_SERVER=OFF"
            "-DLLAMA_BUILD_TOOLS=ON" # mtmd lives under tools/
            "-DLLAMA_BUILD_APP=OFF"
            "-DLLAMA_BUILD_COMMON=ON"
            "-DLLAMA_BUILD_HTML=OFF"
            "-DLLAMA_BUILD_UI=OFF"
            "-DMTMD_VIDEO=OFF" # avoid the optional ffmpeg dependency
          ];
          # Prune `$out/lib` to the core shared libs the sys crate should link (llama/ggml*/mtmd).
          # LLAMA_BUILD_TOOLS=ON installs tool-support byproducts (`*-impl` libs, libcommon) into the
          # same dir; the crate's prebuilt mode globs the dir and links EVERY lib, so anything that is
          # not a core runtime lib must be removed (mirrors the Windows `prebuilt/lib` whitelist).
          # Guard that libmtmd survived so a broken build fails loudly instead of silently dropping
          # multimodal support.
          postInstall = ''
            shopt -s nullglob
            for f in "$out/lib"/*; do
              bn="$(basename "$f")"
              case "$bn" in
                libllama.so* | libggml*.so* | libmtmd.so*) ;;
                *) echo "prune: removing non-core entry $bn from \$out/lib"; rm -rf "$f" ;;
              esac
            done
            if [ -z "$(find "$out/lib" -maxdepth 1 -name 'libmtmd.so*' -print -quit)" ]; then
              echo "FATAL: libmtmd.so missing from \$out/lib after build"; ls -la "$out/lib"; exit 1
            fi
            echo "== llamaCpp: \$out/lib (link-time whitelist) =="; ls -la "$out/lib"
          '';
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
              # Features-matched deps artifact: the heavy native trees this lane pulls in
              # (llama.cpp's cmake build, mistral.rs's candle) are NOT in the default-feature
              # `cargoArtifacts`, so without this every workspace source edit recompiles them from
              # scratch. Building deps-only with the SAME features moves them into a cached layer
              # keyed on Cargo.lock/toolchain/pin — "compile once per arch until the hash changes"
              # (the Windows llama lane's `daemonInferLlamaWindowsDeps` is the same pattern).
              cargoArtifacts = craneLib.buildDepsOnly (
                commonArgs
                // {
                  pname = "daemon-infer-${name}-deps";
                  cargoExtraArgs = "-p daemon-infer --features ${features}";
                  nativeBuildInputs = engineNativeInputs;
                  LIBCLANG_PATH = libclangPath;
                }
              );
              cargoExtraArgs = "-p daemon-infer --features ${features}";
              # The patchelf/libgomp RUNPATH fixup is a Linux-only concern (ELF binaries +
              # libgomp.so). On darwin the worker is a Mach-O binary and llama.cpp's OpenMP is
              # resolved through the Apple toolchain, so both the tool and the postInstall step
              # are gated to Linux (avoids "patchelf: not an ELF executable" on aarch64-darwin).
              nativeBuildInputs = engineNativeInputs ++ lib.optionals pkgs.stdenv.isLinux [ pkgs.patchelf ];
              LIBCLANG_PATH = libclangPath;
              doCheck = false;
              postInstall = lib.optionalString pkgs.stdenv.isLinux ''
                patchelf --add-rpath ${pkgs.gcc.cc.lib}/lib "$out/bin/daemon-infer"
              '';
            }
          );

        # The llama lane ships with multimodal projector loading (`mtmd`): this is a from-source
        # cmake build inside the sandbox, so tools/mtmd compiles in. The dev-shell prebuilt
        # (`packages.llama-cpp`) now also builds tools/mtmd (LLAMA_BUILD_TOOLS=ON) and ships
        # libmtmd, so `cargo build -p daemon-infer --features llama,mtmd,dynamic-link` links it in
        # the dev shell too — dev multimodal matches the sandbox/bundle behavior.
        daemon-infer-llama = buildEngineWorker "llama" "llama,mtmd";
        daemon-infer-mistralrs = buildEngineWorker "mistralrs" "mistralrs";

        # ------------------------------------------------------------------------------------
        # macOS (aarch64-darwin) Metal inference lanes. Metal is the first-class Apple GPU path
        # for both engines and these outputs are darwin-gated (see `packages` below), so Linux
        # evaluation is untouched.
        #
        # Deliberate decision: NO Vulkan/MoltenVK lane on macOS. Metal is the native Apple
        # backend; layering the Vulkan lane through MoltenVK would add a translation shim (and a
        # MoltenVK dependency) for no benefit — the engines target Metal directly and fall back to
        # Metal anyway. The Linux `daemon-infer-vulkan` lane stays the only Vulkan output.
        #
        # Framework linkage: on this modern nixpkgs-darwin the default `apple-sdk` in the stdenv
        # supplies Foundation/Metal/MetalKit/Accelerate to the linker automatically, so these
        # lanes add NO explicit framework/SDK inputs (verified: the four lanes build green on an
        # M1 without them). Add SDK inputs here only if a future configure/link error demands it.

        # llama.cpp Metal lane: `--features llama,mtmd,metal` forwards to `llama-cpp-4/metal` ->
        # `llama-cpp-sys-4` -> `GGML_METAL=ON` in the from-source cmake build. `mtmd` rides the
        # same sandbox cmake build (LLAMA_BUILD_TOOLS) exactly as the Linux llama lane. No xcrun
        # shader step is needed: `GGML_METAL_EMBED_LIBRARY` defaults on from `GGML_METAL`, so the
        # Metal shader library is embedded and JIT-compiled at runtime.
        daemon-infer-metal = buildEngineWorker "metal" "llama,mtmd,metal";

        # mistral.rs Metal lane: `--features mistralrs,mistralrs-metal` forwards to
        # `mistralrs/metal`. `MISTRALRS_METAL_PRECOMPILE=0` forces mistral.rs to JIT-compile its
        # Metal kernels at runtime instead of precompiling them with `xcrun metal` at build time
        # — the xcrun precompile step cannot run in the nix sandbox. Not routed through
        # `buildEngineWorker` because it needs that extra build-time env var; the rest mirrors the
        # Linux mistralrs lane (engine toolchain, LIBCLANG_PATH, doCheck = false, no patchelf).
        daemon-infer-mistralrs-metal = craneLib.buildPackage (
          commonArgs
          // {
            pname = "daemon-infer-mistralrs-metal";
            # Features-matched deps artifact (see `buildEngineWorker`): keeps the mistral.rs/candle
            # tree in a cached layer instead of recompiling it on every workspace source edit.
            cargoArtifacts = craneLib.buildDepsOnly (
              commonArgs
              // {
                pname = "daemon-infer-mistralrs-metal-deps";
                cargoExtraArgs = "-p daemon-infer --features mistralrs,mistralrs-metal";
                nativeBuildInputs = engineNativeInputs;
                LIBCLANG_PATH = libclangPath;
                MISTRALRS_METAL_PRECOMPILE = "0";
              }
            );
            cargoExtraArgs = "-p daemon-infer --features mistralrs,mistralrs-metal";
            nativeBuildInputs = engineNativeInputs;
            LIBCLANG_PATH = libclangPath;
            MISTRALRS_METAL_PRECOMPILE = "0";
            doCheck = false;
          }
        );

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
            # Features-matched deps artifact (see `buildEngineWorker`): moves the from-source
            # Vulkan llama.cpp cmake build into a cached layer instead of recompiling it per edit.
            cargoArtifacts = craneLib.buildDepsOnly (
              commonArgs
              // {
                pname = "daemon-infer-vulkan-deps";
                cargoExtraArgs = "-p daemon-infer --features vulkan";
                nativeBuildInputs = engineNativeInputs ++ [ pkgs.shaderc ];
                buildInputs = [ pkgs.vulkan-headers pkgs.vulkan-loader pkgs.spirv-headers ];
                LIBCLANG_PATH = libclangPath;
              }
            );
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

        # ------------------------------------------------------------------------------------
        # Windows cross lane (x86_64-pc-windows-gnu): daemon.exe + daemon-cli.exe for the NSIS
        # installer to bundle. Deliberately separate outputs — never part of the default gate.
        # The flake's nixpkgs tracks the logos-co `mingw-integration` fork, so `pkgsCross.mingwW64`
        # carries the MinGW fixes this lane relies on. The engine worker lanes (llama/mistralrs)
        # cross too — see the `daemon-infer-*-windows` outputs and `llamaCppWindows` further below.
        pkgsWindows = pkgs.pkgsCross.mingwW64;
        windowsTriple = "x86_64-pc-windows-gnu";
        mingwCc = pkgsWindows.stdenv.cc;
        mingwTargetCc = "${mingwCc}/bin/${mingwCc.targetPrefix}cc";

        # The pinned stable toolchain plus the windows-gnu std, combined per fenix's cross recipe
        # (same stable channel as `rustToolchain`, so host and cross rustc stay in lockstep).
        rustToolchainWindows = fenix.packages.${system}.combine [
          rustToolchain
          fenix.packages.${system}.targets.${windowsTriple}.stable.rust-std
        ];

        craneLibWindows = (crane.mkLib pkgs).overrideToolchain rustToolchainWindows;

        windowsCommonArgs = commonArgs // {
          CARGO_BUILD_TARGET = windowsTriple;
          CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER = mingwTargetCc;
          # Fully static mingw runtime (libgcc + winpthread folded into the exe) so the shipped
          # artifact depends on system DLLs only — nothing to place next to it in the installer.
          CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUSTFLAGS = "-C link-arg=-static -C link-arg=-static-libgcc -L ${pkgsWindows.windows.pthreads}/lib";
          # cc-rs–built C deps (ring, bundled sqlite, secp256k1, aws-lc) target compiler/archiver.
          TARGET_CC = mingwTargetCc;
          TARGET_AR = "${mingwCc.bintools}/bin/${mingwCc.targetPrefix}ar";
          # aws-lc-sys (rustls' aws-lc-rs provider, via daemon-host TLS): assemble from source with
          # the real nasm — its prebuilt-NASM helper script does not run under the nix sandbox
          # (unpatched shebang; see crane's cross-compiling-aws-lc-sys FAQ).
          AWS_LC_SYS_PREBUILT_NASM = 0;
          # aws-lc compiles with -Werror; these gcc-15/mingw false positives would fail the build.
          CFLAGS = "-Wno-stringop-overflow -Wno-array-bounds -Wno-restrict";
          # mingw pthread headers for the windows-targeted C compiles (aws-lc feature probes).
          # Passed as path strings (here and in the -L above) rather than as buildInputs: nixpkgs
          # splicing would try to re-resolve the windows-only package for the linux hostPlatform
          # and refuse to evaluate it.
          CFLAGS_x86_64_pc_windows_gnu = "-I${pkgsWindows.windows.pthreads}/include";
          # Windows artifacts cannot run in the linux build sandbox; wine smoke is manual.
          doCheck = false;
          nativeBuildInputs = [
            pkgs.nasm
            pkgs.cmake
          ];
          depsBuildBuild = [ mingwCc ];
        };

        # Dependency-only artifacts for the two shipped bins. Scoped with `-p` so the unix-only
        # dev-deps (the `nix` signal crate) and test-only trees never compile for windows.
        windowsCargoArtifacts = craneLibWindows.buildDepsOnly (
          windowsCommonArgs
          // {
            pname = "daemon-workspace-windows";
            cargoExtraArgs = "-p daemon -p daemon-cli";
          }
        );

        buildWindowsPackage =
          packageName:
          craneLibWindows.buildPackage (
            windowsCommonArgs
            // {
              pname = "${packageName}-windows";
              version = baseVersion;
              cargoArtifacts = windowsCargoArtifacts;
              cargoExtraArgs = "-p ${packageName}";
              # Same reproducible build-metadata injection as the native packages.
              DAEMON_BUILD_ID = buildId;
            }
          );

        daemon-windows = buildWindowsPackage "daemon";
        daemon-cli-windows = buildWindowsPackage "daemon-cli";

        # ------------------------------------------------------------------------------------
        # Windows inference-engine worker lanes (x86_64-pc-windows-gnu): the `daemon-infer` worker
        # built with the llama.cpp and mistral.rs engines for the NSIS installer to bundle. Like
        # the daemon/daemon-cli windows lanes these are deliberately separate outputs, never part
        # of the default gate.
        #
        # Windows llama.cpp rides upstream's dynamic-backend model (`GGML_BACKEND_DL`): the worker
        # links ONLY the core libraries (llama/ggml/ggml-base/mtmd) while the compute backends are
        # runtime-loadable modules (`ggml-cpu.dll`, `ggml-vulkan.dll`) that `ggml_backend_load_all()`
        # — called from `llama_backend_init()` inside libllama — discovers beside the exe. On a
        # GPU-less machine `ggml-vulkan.dll` (or its `vulkan-1.dll` dependency) simply fails to load
        # and ggml falls back to the always-present CPU backend. The crate's from-source Vulkan path
        # is NOT used here (its `find_vulkan_sdk_windows()` wants an MSVC SDK layout and hard-links
        # `vulkan-1`) — the `vulkan` cargo feature must never be enabled on windows.
        #
        # Import contract (enforced by the objdump guards in the llama lane's postInstall):
        #   daemon-infer.exe imports llama/ggml/ggml-base(/mtmd) — NOT ggml-vulkan, NOT vulkan-1.
        #   ggml-vulkan.dll ships beside the exe and imports vulkan-1.dll (runtime GPU backend).

        # Host (build-platform) toolchain file for ggml-vulkan's `vulkan-shaders-gen`: that helper
        # is compiled and RUN during the build to emit the SPIR-V shader headers, so it must target
        # the build machine, not the mingw target. This is upstream's documented cross escape hatch
        # (`GGML_VULKAN_SHADERS_GEN_TOOLCHAIN` in ggml/src/ggml-vulkan/CMakeLists.txt).
        llamaVulkanShaderHostToolchain = pkgs.writeText "llama-vulkan-shader-host-toolchain.cmake" ''
          set(CMAKE_SYSTEM_NAME Linux)
          set(CMAKE_C_COMPILER ${pkgs.stdenv.cc}/bin/cc)
          set(CMAKE_CXX_COMPILER ${pkgs.stdenv.cc}/bin/c++)
          set(CMAKE_AR ${pkgs.binutils}/bin/ar)
          set(CMAKE_RANLIB ${pkgs.binutils}/bin/ranlib)
          set(CMAKE_FIND_ROOT_PATH "")
          set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
          set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY NEVER)
          set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE NEVER)
          set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE NEVER)
        '';

        # Mingw target C++ compiler + objdump (the latter inspects the produced PE import tables).
        mingwCxx = "${mingwCc}/bin/${mingwCc.targetPrefix}c++";
        mingwObjdump = "${mingwCc.bintools}/bin/${mingwCc.targetPrefix}objdump";

        # The mingw C++/gcc runtime DLLs the llama.cpp DLLs dynamically depend on (a fully-`-static`
        # DLL build hits `_Unwind_Resume` multiple-definition across the interdependent ggml DLLs).
        # The thread runtime (libmcfgthread-2.dll — this fork's model, not winpthread) and the vulkan
        # loader (vulkan-1.dll) are added automatically by the nixpkgs mingw DLL-fixup phase. The
        # worker exe itself is statically linked and does not import any of these.
        mingwRuntimeDlls = [
          "${mingwCc.cc.lib}/x86_64-w64-mingw32/lib/libgcc_s_seh-1.dll"
          "${mingwCc.cc.lib}/x86_64-w64-mingw32/lib/libstdc++-6.dll"
        ];

        # The nixpkgs mingw gcc ships no `libgomp` for the windows target, yet llama-cpp-sys-4's
        # prebuilt path unconditionally emits `cargo:rustc-link-lib=gomp` for windows-gnu (the
        # `openmp` cargo feature is forced on `llama-cpp-4` in the workspace Cargo.toml, and cannot
        # be turned off per-consumer). Because the llama.cpp DLLs are built `GGML_OPENMP=OFF`, no
        # OpenMP symbol is ever referenced, so an EMPTY `libgomp.a` satisfies `-lgomp` with zero
        # runtime effect (the worker gains no libgomp DLL dependency).
        mingwGompStub = pkgs.runCommandLocal "mingw-libgomp-stub" { } ''
          mkdir -p "$out/lib"
          "${mingwCc.bintools}/bin/${mingwCc.targetPrefix}ar" crs "$out/lib/libgomp.a"
        '';

        # From-source, shared-lib, DL-backend build of the pinned llama.cpp commit for windows-gnu.
        # cmake is cross-configured by the mingw stdenv automatically. The compute backends build as
        # MODULE libraries (upstream `ggml_add_backend_library` under `GGML_BACKEND_DL`), so they
        # emit a `.dll` with NO import lib and install to bin — they are physically un-linkable and
        # can only be loaded at runtime, which is exactly the model we want. All DLLs land in
        # `$out/bin` (the set shipped beside the exe) and, mirrored, `$out/prebuilt/bin`; headers in
        # `$out/prebuilt/include`; and the core import libs (llama/ggml/ggml-base/mtmd) in
        # `$out/prebuilt/lib` with any *vulkan* import lib stripped as a belt-and-suspenders guard.
        llamaCppWindows = pkgsWindows.stdenv.mkDerivation {
          pname = "llama-cpp-windows-prebuilt";
          version = "b9496";
          src = llama-cpp-src;
          strictDeps = true;
          # Host build tools: cmake/ninja drive the cross build; shaderc's `glslc` and the host gcc
          # (via the toolchain file above) build+run the SPIR-V shader generator on the build host.
          nativeBuildInputs = [ pkgs.cmake pkgs.ninja pkgs.pkg-config pkgs.shaderc ];
          # Target (windows) libraries the vulkan backend links/finds.
          buildInputs = [ pkgsWindows.vulkan-loader pkgsWindows.vulkan-headers pkgsWindows.spirv-headers ];
          cmakeFlags = [
            "-DBUILD_SHARED_LIBS=ON"
            "-DGGML_BACKEND_DL=ON"
            "-DGGML_VULKAN=ON"
            "-DGGML_OPENMP=OFF"
            "-DGGML_NATIVE=OFF"
            "-DLLAMA_BUILD_COMMON=ON"
            "-DLLAMA_BUILD_TOOLS=ON" # mtmd lives under tools/
            "-DLLAMA_BUILD_APP=OFF"
            "-DLLAMA_BUILD_TESTS=OFF"
            "-DLLAMA_BUILD_EXAMPLES=OFF"
            "-DLLAMA_BUILD_SERVER=OFF"
            "-DLLAMA_CURL=OFF"
            "-DMTMD_VIDEO=OFF" # avoid the optional ffmpeg dependency
            "-DVulkan_INCLUDE_DIR=${pkgsWindows.vulkan-headers}/include"
            "-DVulkan_LIBRARY=${pkgsWindows.vulkan-loader}/lib/libvulkan-1.dll.a"
            "-DVulkan_GLSLC_EXECUTABLE=${pkgs.shaderc.bin}/bin/glslc"
            "-DSPIRV-Headers_DIR=${pkgsWindows.spirv-headers}/share/cmake/SPIRV-Headers"
            "-DGGML_VULKAN_SHADERS_GEN_TOOLCHAIN=${llamaVulkanShaderHostToolchain}"
          ];
          # Runtime linkage is intentionally the mingw default (dynamic libgcc/libstdc++/mcfgthread):
          # a `-static` DLL build makes each interdependent ggml DLL statically embed AND auto-export
          # libgcc's exception-unwinding symbols, which then collide (multiple definition of
          # `_Unwind_Resume`) when a downstream DLL links an upstream one. Instead the mingw runtime
          # DLLs ship in the runtime set (see postInstall) and `vulkan-1.dll` stays a dynamic import.
          postInstall = ''
            mkdir -p "$out/prebuilt/bin" "$out/prebuilt/lib" "$out/prebuilt/include"

            # The mingw C++/gcc/winpthread runtime DLLs the llama.cpp DLLs depend on.
            for rt in ${lib.concatStringsSep " " mingwRuntimeDlls}; do
              cp -f "$rt" "$out/bin/"
            done

            # Public headers for bindgen (the crate's build.rs also -I's the vendored source, so
            # this is a convenience mirror, not strictly required).
            if [ -d "$out/include" ]; then
              cp -r "$out/include/." "$out/prebuilt/include/"
            fi

            # Runtime DLLs: the installed set in $out/bin (core libs + DL backend modules). Belt and
            # suspenders: pull in any DLL from the build tree that install() may have missed.
            for dll in $(find "$PWD" "$out" -name '*.dll' -type f); do
              bn="$(basename "$dll")"
              if [ ! -e "$out/bin/$bn" ]; then
                cp "$dll" "$out/bin/$bn"
              fi
            done
            cp "$out/bin/"*.dll "$out/prebuilt/bin/"

            # Import libs for the crate's link step: ONLY the core shared libs the worker links
            # against (llama/ggml/ggml-base/mtmd). The crate's prebuilt mode globs *.a in this dir
            # and links EVERY one, so the LLAMA_BUILD_TOOLS impl import libs (llama-bench-impl, ...)
            # and any *vulkan* import lib must be kept out — otherwise the worker would bind tool /
            # GPU DLLs at link time. The DL backends (ggml-cpu/ggml-vulkan) are MODULE libs with no
            # import lib and load purely at runtime. Import-lib names carry the mingw `lib` prefix
            # (libggml.dll.a) even where the DLL does not (ggml.dll); accept either form.
            for base in llama ggml ggml-base mtmd; do
              for cand in "$out/lib/lib$base.dll.a" "$out/lib/$base.dll.a"; do
                if [ -e "$cand" ]; then cp "$cand" "$out/prebuilt/lib/"; fi
              done
            done

            # Guards: the DL backend + core DLLs must exist; no vulkan import lib may remain. DLL
            # names may or may not carry the mingw `lib` prefix (ggml libs drop it, llama/mtmd keep
            # it), so accept either form. Use `find` (not glob-in-`ls`) because nix builders enable
            # `nullglob`, under which an unmatched `ls *pat*` would list the cwd and false-fire.
            have_dll() { [ -e "$out/bin/$1.dll" ] || [ -e "$out/bin/lib$1.dll" ]; }
            for core in ggml ggml-base ggml-cpu ggml-vulkan llama mtmd; do
              have_dll "$core" || { echo "FATAL: core/backend DLL '$core' missing from \$out/bin"; ls -la "$out/bin"; exit 1; }
            done
            if [ -n "$(find "$out/prebuilt/lib" -iname '*vulkan*' 2>/dev/null)" ]; then
              echo "FATAL: a vulkan import lib leaked into prebuilt/lib"; ls -la "$out/prebuilt/lib"; exit 1
            fi
            { [ -e "$out/prebuilt/lib/llama.dll.a" ] || [ -e "$out/prebuilt/lib/libllama.dll.a" ]; } \
              || { echo "FATAL: llama import lib missing from prebuilt/lib"; ls -la "$out/prebuilt/lib"; exit 1; }
            echo "== llamaCppWindows: \$out/prebuilt/lib (link-time whitelist) =="; ls -la "$out/prebuilt/lib"

            echo "== llamaCppWindows: \$out/bin =="; ls -la "$out/bin"
            echo "== llamaCppWindows: \$out/prebuilt/lib =="; ls -la "$out/prebuilt/lib"
          '';
        };

        # Convenience references to the prebuilt tree (single-output derivation, so its store path
        # is `${llamaCppWindows}`). Consumed as `LLAMA_PREBUILT_DIR` + the runtime DLL source.
        llamaCppWindowsPrebuiltDir = "${llamaCppWindows}/prebuilt";
        llamaCppWindowsRuntimeDir = "${llamaCppWindows}/bin";

        # Clang args for bindgen when it cross-targets windows-gnu with the HOST libclang: point it
        # at the mingw C runtime headers and the mingw libstdc++ headers (clang supplies its own
        # stddef/stdint builtins). Shared by every engine lane that runs bindgen (llama-cpp-sys-4,
        # onig_sys). The llama lane appends the prebuilt include dir on top of this.
        windowsBindgenClangArgs = lib.concatStringsSep " " [
          "--target=x86_64-w64-windows-gnu"
          "-isystem ${mingwCc.cc}/include/c++/${mingwCc.cc.version}"
          "-isystem ${mingwCc.cc}/include/c++/${mingwCc.cc.version}/x86_64-w64-mingw32"
          "-isystem ${mingwCc.cc}/include/c++/${mingwCc.cc.version}/backward"
          "-isystem ${mingwCc.libc.dev}/include"
          # This fork's libstdc++ uses the mcfgthread threading model; its <bits/gthr-default.h>
          # includes <mcfgthread/gthr.h>, which lives in the mcfgthread dev headers.
          "-isystem ${pkgsWindows.windows.mcfgthreads.dev}/include"
        ];

        # Shared cross env for the engine worker lanes: the base windows cross args plus bindgen
        # (host libclang), the mtp_shim C++ compiler (cc-rs), and the empty-libgomp `-L` so the
        # forced `-lgomp` resolves. Target-scoped compiler vars are preferred over global CC/CXX so
        # host build scripts keep using the native toolchain.
        windowsEngineCommonArgs = windowsCommonArgs // {
          nativeBuildInputs = (windowsCommonArgs.nativeBuildInputs or [ ]) ++ [
            pkgs.ninja
            pkgs.pkg-config
            pkgs.llvmPackages.libclang.lib
          ];
          # mtp_shim is C++ (compiled by cc-rs); onig_sys / ring / etc. are C (TARGET_CC already set
          # in windowsCommonArgs). Scope C++ to the target triple; leave host CXX alone.
          TARGET_CXX = mingwCxx;
          "CC_x86_64_pc_windows_gnu" = mingwTargetCc;
          "CXX_x86_64_pc_windows_gnu" = mingwCxx;
          LIBCLANG_PATH = libclangPath;
          "BINDGEN_EXTRA_CLANG_ARGS_x86_64_pc_windows_gnu" = windowsBindgenClangArgs;
          # Extend the base static RUSTFLAGS (winpthread -L from windowsCommonArgs) with:
          #  * the empty-libgomp stub dir (resolves the forced `-lgomp`);
          #  * the mingw gcc target lib dir, so rustc can find `libstdc++.a` for the prebuilt path's
          #    `cargo:rustc-link-lib=static=stdc++` (rustc validates `static=` libs against its own
          #    -L set, unlike the gcc driver's implicit search dirs).
          CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUSTFLAGS =
            windowsCommonArgs.CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUSTFLAGS
            + " -L ${mingwGompStub}/lib"
            + " -L ${mingwCc.cc}/x86_64-w64-mingw32/lib";
        };

        # The llama lane needs the prebuilt shared llama.cpp (skips cmake in build.rs) and the
        # prebuilt include dir on bindgen's search path.
        windowsLlamaEngineArgs = windowsEngineCommonArgs // {
          LLAMA_PREBUILT_DIR = llamaCppWindowsPrebuiltDir;
          LLAMA_PREBUILT_SHARED = "1";
          "BINDGEN_EXTRA_CLANG_ARGS_x86_64_pc_windows_gnu" =
            windowsBindgenClangArgs + " -I${llamaCppWindowsPrebuiltDir}/include";
        };

        daemonInferLlamaWindowsDeps = craneLibWindows.buildDepsOnly (
          windowsLlamaEngineArgs
          // {
            pname = "daemon-infer-llama-windows-deps";
            # `default` features are empty; match the native lane's convention (no
            # --no-default-features). The target comes from CARGO_BUILD_TARGET.
            cargoExtraArgs = "-p daemon-infer --features llama,mtmd";
          }
        );

        daemon-infer-llama-windows = craneLibWindows.buildPackage (
          windowsLlamaEngineArgs
          // {
            pname = "daemon-infer-llama-windows";
            version = baseVersion;
            cargoArtifacts = daemonInferLlamaWindowsDeps;
            cargoExtraArgs = "-p daemon-infer --features llama,mtmd";
            DAEMON_BUILD_ID = buildId;
            postInstall = ''
              # Ship the worker's runtime DLL closure beside the exe: the core libs + DL backends
              # (incl. ggml-vulkan.dll) + mingw/mcfgthread runtime + vulkan-1.dll. Skip the
              # LLAMA_BUILD_TOOLS byproducts (`*-impl.dll`, `libllama-common.dll`) — nothing the
              # worker loads imports them (verified via objdump). `llama-cpp-windows` keeps the full
              # set; only the shipped worker set is trimmed.
              for dll in ${llamaCppWindowsRuntimeDir}/*.dll; do
                bn="$(basename "$dll")"
                case "$bn" in
                  *-impl.dll | libllama-common.dll) continue ;;
                esac
                cp "$dll" "$out/bin/"
              done

              objdump="${mingwObjdump}"
              exe="$out/bin/daemon-infer.exe"
              test -e "$exe" || { echo "FATAL: daemon-infer.exe not installed"; ls -la "$out/bin"; exit 1; }

              echo "== daemon-infer.exe imports =="
              "$objdump" -p "$exe" | grep -i 'DLL Name:' || true

              # Guard 1: the worker imports NEITHER vulkan-1.dll NOR any ggml-vulkan dll.
              if "$objdump" -p "$exe" | grep -qi 'vulkan-1\.dll'; then
                echo "FATAL: daemon-infer.exe imports vulkan-1.dll (must be runtime-DL only)"; exit 1
              fi
              if "$objdump" -p "$exe" | grep -qi 'ggml-vulkan'; then
                echo "FATAL: daemon-infer.exe imports a ggml-vulkan dll (must be runtime-DL only)"; exit 1
              fi

              # Guard 2: the ggml-vulkan backend DLL must ship AND import vulkan-1.dll (the GPU
              # backend; on a GPU-less host it just fails to load -> CPU fallback). Name may or may
              # not carry the mingw `lib` prefix.
              vk="$(find "$out/bin" -iname '*ggml-vulkan*.dll' 2>/dev/null | head -1)"
              [ -n "$vk" ] || { echo "FATAL: ggml-vulkan backend DLL missing from \$out/bin"; ls -la "$out/bin"; exit 1; }
              if ! "$objdump" -p "$vk" | grep -qi 'vulkan-1\.dll'; then
                echo "FATAL: $(basename "$vk") does not import vulkan-1.dll"; "$objdump" -p "$vk" | grep -i 'DLL Name:'; exit 1
              fi

              # Guard 3: the core runtime DLLs are present beside the exe (prefix-agnostic).
              have_dll() { [ -e "$out/bin/$1.dll" ] || [ -e "$out/bin/lib$1.dll" ]; }
              for core in ggml ggml-base ggml-cpu ggml-vulkan llama mtmd; do
                have_dll "$core" || { echo "FATAL: core DLL '$core' missing"; ls -la "$out/bin"; exit 1; }
              done

              echo "== per-DLL imports (runtime dependency map for packaging) =="
              for d in "$out/bin/"*.dll; do
                echo "--- $(basename "$d") ---"
                "$objdump" -p "$d" | grep -i 'DLL Name:' || true
              done
              echo "import-contract guards passed"
            '';
          }
        );

        # mistral.rs on windows is CPU-only (candle has no vulkan backend). Its only new native dep
        # is `onig_sys` (C, covered by TARGET_CC + the bindgen clang args). No prebuilt env.
        daemonInferMistralrsWindowsDeps = craneLibWindows.buildDepsOnly (
          windowsEngineCommonArgs
          // {
            pname = "daemon-infer-mistralrs-windows-deps";
            cargoExtraArgs = "-p daemon-infer --features mistralrs";
          }
        );

        daemon-infer-mistralrs-windows = craneLibWindows.buildPackage (
          windowsEngineCommonArgs
          // {
            pname = "daemon-infer-mistralrs-windows";
            version = baseVersion;
            cargoArtifacts = daemonInferMistralrsWindowsDeps;
            cargoExtraArgs = "-p daemon-infer --features mistralrs";
            DAEMON_BUILD_ID = buildId;
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
            # Features-matched deps artifact (see `buildEngineWorker`): `hyperon` is a big non-default
            # git dep absent from the default-feature `cargoArtifacts`, so without this it recompiles
            # on every workspace source edit. Deps-only with `--features hyperon` caches it.
            cargoArtifacts = craneLib.buildDepsOnly (
              commonArgs
              // {
                pname = "daemon-metta-deps";
                cargoExtraArgs = "-p daemon-metta --features hyperon";
                nativeBuildInputs = [ pkgs.pkg-config ];
              }
            );
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
            daemon-windows
            daemon-cli-windows
            daemon-infer-llama-windows
            daemon-infer-mistralrs-windows
            ;
          # Prebuilt llama.cpp (shared, CPU + Vulkan) matching the crate's vendored commit; consumed
          # by the dev shells via `LLAMA_PREBUILT_DIR` to compile the llama lane without cmake.
          llama-cpp = llamaCpp;
          # Prebuilt shared llama.cpp for windows-gnu (DL backends: CPU + Vulkan). Consumed by the
          # `daemon-infer-llama-windows` lane and available for the superproject's NSIS bundling.
          llama-cpp-windows = llamaCppWindows;
          default = daemon;
        }
        # macOS-only Metal inference lanes. Darwin-gated (mirroring how the devShells gate the
        # Linux-only vulkan/cuda shells) so `nix flake show` / eval on Linux never lists or forces
        # them; they only appear under `packages.aarch64-darwin`.
        // lib.optionalAttrs pkgs.stdenv.isDarwin {
          inherit
            daemon-infer-metal
            daemon-infer-mistralrs-metal
            ;
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
            # sandbox), so `cargo build -p daemon-infer --features llama,mtmd,dynamic-link` links that
            # prebuilt and skips cmake entirely (only the `cc`-built `mtp_shim` compiles locally). The
            # prebuilt now ships libmtmd (built with LLAMA_BUILD_TOOLS=ON), so the `mtmd` multimodal
            # feature links in the dev shell exactly as it does in the sandbox lane / bundle.
            # That prebuilt also bundles the Vulkan backend (one CPU+Vulkan artifact), so
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
                pkgs.python3 # execute_code tool: the child interpreter for its subprocess tests
                pkgs.bubblewrap # execute_code tool: the OS sandbox (bwrap); tests guard on usability
              ]
              ++ engineNativeInputs;
          };
        }
        // {
          # Interactive iteration on the windows cross lane: `cargo build -p daemon` in here
          # targets x86_64-pc-windows-gnu with the same toolchain/env as `packages.daemon-windows`.
          windows-cross = pkgs.mkShell {
            CARGO_BUILD_TARGET = windowsTriple;
            CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER = mingwTargetCc;
            CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUSTFLAGS = "-C link-arg=-static -C link-arg=-static-libgcc -L ${pkgsWindows.windows.pthreads}/lib";
            TARGET_CC = mingwTargetCc;
            TARGET_AR = "${mingwCc.bintools}/bin/${mingwCc.targetPrefix}ar";
            AWS_LC_SYS_PREBUILT_NASM = 0;
            CFLAGS = "-Wno-stringop-overflow -Wno-array-bounds -Wno-restrict";
            CFLAGS_x86_64_pc_windows_gnu = "-I${pkgsWindows.windows.pthreads}/include";
            SSL_CERT_FILE = caBundle;
            NIX_SSL_CERT_FILE = caBundle;
            packages = [
              rustToolchainWindows
              mingwCc
              pkgs.nasm
              pkgs.cmake
            ];
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
