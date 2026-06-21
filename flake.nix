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
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      crane,
      fenix,
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

        rustToolchain = fenix.packages.${system}.stable.withComponents [
          "cargo"
          "clippy"
          "rust-src"
          "rustc"
          "rustfmt"
        ];

        craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;

        src = craneLib.cleanCargoSource ./.;

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
            # Worker engine toolchain (cmake/clang/libclang) is present so a dev can build an engine
            # lane locally (`cargo build -p daemon-infer --features llama`). The default
            # `cargo test --workspace` still builds only the stub worker — no engine, no cmake step.
            #
            # Build-matrix shrinking: llama-cpp-4 exposes a `prebuilt` feature that links a pre-built
            # llama.cpp from $LLAMA_PREBUILT_DIR instead of compiling it here — point that at a cached
            # build to drop the per-lane cmake compile in CI.
            LIBCLANG_PATH = libclangPath;
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
