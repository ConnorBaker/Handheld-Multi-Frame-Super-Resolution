{
  inputs = {
    flake-parts = {
      inputs.nixpkgs-lib.follows = "nixpkgs";
      url = "github:hercules-ci/flake-parts";
    };

    # nixpkgs.url = "github:NixOS/nixpkgs";
    nixpkgs.url = "github:NixOS/nixpkgs/pull/462810/merge";

    git-hooks-nix = {
      inputs.nixpkgs.follows = "nixpkgs";
      url = "github:cachix/git-hooks.nix";
    };

    treefmt-nix = {
      inputs.nixpkgs.follows = "nixpkgs";
      url = "github:numtide/treefmt-nix";
    };
  };

  outputs =
    inputs:
    let
      inherit (inputs.flake-parts.lib) mkFlake;
      systems = [
        "aarch64-linux"
        "x86_64-linux"
      ];
    in
    mkFlake { inherit inputs; } {
      inherit systems;

      imports = [
        inputs.treefmt-nix.flakeModule
        inputs.git-hooks-nix.flakeModule
      ];

      flake.overlays.default = import ./overlay.nix;

      perSystem =
        {
          config,
          lib,
          pkgs,
          system,
          ...
        }:
        {
          _module.args.pkgs = import inputs.nixpkgs {
            config = {
              allowUnfree = true;
              cudaSupport = true;
              # Support Orin and Ada
              cudaCapabilities = if system == "aarch64-linux" then [ "8.7" ] else [ "8.9" ];
            };
            localSystem = { inherit system; };
            overlays = [ inputs.self.overlays.default ];
          };

          legacyPackages = pkgs;

          devShells.default =
            let
              inherit (pkgs)
                mkShell
                python3
                cudaPackages
                addDriverRunpath
                ;
            in
            mkShell {
              packages = [
                (python3.withPackages (
                  ps: with ps; [
                    exifread
                    helion
                    imageio
                    jaxtyping
                    matplotlib
                    numba
                    numpy
                    omegaconf
                    opencv4
                    optuna
                    rawpy
                    scipy
                    scikit-image
                    torch
                    torchmetrics
                    torchvision
                    tqdm
                  ]
                ))
                pkgs.nodePackages_latest.nodejs
                # For libnvvm.so, libcudart.so needed by numba
                cudaPackages.cuda_nvcc
                cudaPackages.cuda_cudart
                pkgs.exiftool
              ];
              shellHook = ''
                export LD_LIBRARY_PATH=${
                  lib.makeLibraryPath [
                    addDriverRunpath.driverLink
                    "${cudaPackages.cuda_nvcc}/nvvm"
                    cudaPackages.cuda_cudart
                  ]
                }
              '';
            };

          pre-commit.settings.hooks = {
            # Formatter checks
            treefmt = {
              enable = true;
              package = config.treefmt.build.wrapper;
            };

            # Nix checks
            deadnix.enable = true;
            nil.enable = true;
            statix.enable = true;
          };

          treefmt = {
            projectRootFile = "flake.nix";
            programs.nixfmt.enable = true;
          };
        };
    };
}
