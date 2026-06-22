{
  description = "daemon Rust workspace";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
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

        # A from-source, CPU, shared-lib build of the exact llama.cpp commit that `llama-cpp-sys-4`
        # 0.3.2 vendors. cmake runs in the Nix build sandbox (which has `/bin/sh`), so this sidesteps
        # the missing-`/bin/sh` trap. Pointing `LLAMA_PREBUILT_DIR` at it lets a plain `cargo build
        # --features llama,dynamic-link` skip cmake entirely in the dev shell (only the `cc`-built
        # `mtp_shim` compiles locally). `GGML_NATIVE=OFF` keeps the store path portable/reproducible.
        llamaCpp = pkgs.stdenv.mkDerivation {
          pname = "llama-cpp-prebuilt";
          version = "b9496";
          src = llama-cpp-src;
          nativeBuildInputs = [ pkgs.cmake pkgs.ninja pkgs.pkg-config ];
          # Build only the ggml + libllama shared libraries; everything else (tools, the unified
          # `app` binary, server, examples, tests, common, web UI) is dropped — we only consume the
          # `.so`s, and those extra targets pull in deps that fail to link in this trimmed build.
          cmakeFlags = [
            "-DBUILD_SHARED_LIBS=ON"
            "-DGGML_NATIVE=OFF"
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
            (craneLib.filterCargoSources path type) || (builtins.match ".*\\.(gbnf|lark)$" path != null);
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
          version = "0.0.0";
          strictDeps = true;
        };

        cargoArtifacts = craneLib.buildDepsOnly commonArgs;

        buildWorkspacePackage =
          packageName:
          craneLib.buildPackage (
            commonArgs
            // {
              pname = packageName;
              version = "0.0.0";
              inherit cargoArtifacts;
              cargoExtraArgs = "-p ${packageName}";
            }
          );

        daemon = buildWorkspacePackage "daemon";
        daemon-cli = buildWorkspacePackage "daemon-cli";

        # Engine-lane compile checks: build the `daemon-infer` worker with an engine feature so the
        # llama-cpp-4 / mistral.rs glue is type-checked against the real native APIs. These compile
        # C/C++ via cmake, which needs `/bin/sh` + a full stdenv — provided by the nix build sandbox
        # (a raw `cargo` build in the dev shell can't satisfy make's hardcoded `/bin/sh` on NixOS).
        # They are deliberately separate outputs, NOT part of the default workspace gate.
        buildEngineWorker =
          features:
          craneLib.buildPackage (
            commonArgs
            // {
              pname = "daemon-infer-${features}";
              inherit cargoArtifacts;
              cargoExtraArgs = "-p daemon-infer --features ${features}";
              nativeBuildInputs = engineNativeInputs;
              LIBCLANG_PATH = libclangPath;
              doCheck = false;
            }
          );

        daemon-infer-llama = buildEngineWorker "llama";
        daemon-infer-mistralrs = buildEngineWorker "mistralrs";

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
            version = "0.0.0";
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
            daemon-metta
            ;
          # Prebuilt llama.cpp (shared, CPU) matching the crate's vendored commit; consumed by the dev
          # shell via `LLAMA_PREBUILT_DIR` to compile the llama lane without cmake.
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
        };

        devShells = {
          default = craneLib.devShell {
            # Worker engine toolchain (clang/libclang for bindgen, cmake for the GPU lanes) is present
            # so a dev can build an engine lane locally. The default `cargo test --workspace` still
            # builds only the stub worker — no engine, no cmake step.
            #
            # llama lane in the dev shell: this host has no `/bin/sh`, so compiling llama.cpp from
            # source (cmake -> make/ninja, both of which exec `/bin/sh`) fails here. Instead we point
            # `LLAMA_PREBUILT_DIR` at the pinned `packages.llama-cpp` (built from source in the Nix
            # sandbox), so `cargo build -p daemon-infer --features llama,dynamic-link` links that
            # prebuilt and skips cmake entirely (only the `cc`-built `mtp_shim` compiles locally).
            # `LD_LIBRARY_PATH` makes the shared llama/ggml libs + libgomp resolvable when the worker
            # runs in-shell (e.g. `cargo test`, `cargo run`).
            LIBCLANG_PATH = libclangPath;
            LLAMA_PREBUILT_DIR = "${llamaCpp}";
            LLAMA_PREBUILT_SHARED = "1";
            LD_LIBRARY_PATH = "${llamaCpp}/lib:${pkgs.gcc.cc.lib}/lib";
            packages =
              [
                rustToolchain
                fenix.packages.${system}.rust-analyzer
                pkgs.rust-cbindgen
              ]
              ++ engineNativeInputs
              ++ lib.optionals (lib.hasAttr "cargo-deny" pkgs) [
                pkgs.cargo-deny
              ]
              ++ lib.optionals (lib.hasAttr "cargo-nextest" pkgs) [
                pkgs.cargo-nextest
              ];
          };
        }
        # Optional GPU lanes for building/exercising the worker with an accelerated backend
        # (`cargo build -p daemon-infer --features cuda` / `--features vulkan`). Linux-only; the CUDA
        # shell needs an unfree-permitting nixpkgs, so it is built lazily and only on Linux.
        // lib.optionalAttrs pkgs.stdenv.isLinux {
          vulkan = craneLib.devShell {
            LIBCLANG_PATH = libclangPath;
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
              packages =
                [ rustToolchain pkgs.rust-cbindgen ]
                ++ engineNativeInputs
                ++ [ cudaPkgs.cudatoolkit ];
            };
        };
      }
    );
}
