{
  description = "Graph-to-voxel MCP development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python312
            uv
            git
            # C/Fortran toolchain for scipy, numpy, LoopStructural compiled extensions
            gcc
            gfortran
            pkg-config
            openblas
            # VTK for pyvista (optional viz)
            vtk
            # zlib for zarr/numpy
            zlib
          ];

          env = {
            # Prevent uv from downloading its own Python (use Nix-provided one)
            UV_PYTHON_DOWNLOADS = "never";
            UV_PYTHON = "${pkgs.python312}/bin/python3.12";
          };

          shellHook = ''
            # Ensure compiled extensions can find libstdc++ and system libs
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc
              pkgs.openblas
              pkgs.zlib
            ]}:''${LD_LIBRARY_PATH:-}
            echo "graph-to-voxel dev shell ready — run: uv sync --extra dev --extra engines"
          '';
        };
      }
    );
}
