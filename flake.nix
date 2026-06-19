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

        rustToolchain = fenix.packages.${system}.stable.withComponents [
          "cargo"
          "clippy"
          "rust-src"
          "rustc"
          "rustfmt"
        ];

        craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;

        src = craneLib.cleanCargoSource ./.;

        commonArgs = {
          inherit src;
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
      in
      {
        packages = {
          inherit daemon daemon-cli;
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

        devShells.default = craneLib.devShell {
          packages =
            [
              rustToolchain
              fenix.packages.${system}.rust-analyzer
            ]
            ++ lib.optionals (lib.hasAttr "cargo-deny" pkgs) [
              pkgs.cargo-deny
            ]
            ++ lib.optionals (lib.hasAttr "cargo-nextest" pkgs) [
              pkgs.cargo-nextest
            ];
        };
      }
    );
}
