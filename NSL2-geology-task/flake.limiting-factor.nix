{
  description = "negative-space-learning dev environment (limiting-factor: native NixOS, 2x RTX 4090)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
        };
      };
      # Pip-installed wheels (torch, unsloth, triton, ...) dlopen these at runtime.
      wheelRuntimeLibs = with pkgs; [
        stdenv.cc.cc.lib
        zlib
      ];
    in {
      # Docker GPU passthrough (`--device nvidia.com/gpu=all`) requires the
      # NVIDIA Container Toolkit's CDI spec at the OS level — see
      # `hardware.nvidia-container-toolkit.enable` in the `limiting-factor`
      # NixOS config (hosts/workstation/gpu.nix). Not configurable from this
      # devshell.
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          python312
          uv
          cacert

          git
          gcc
          pkg-config
          cmake
          ninja
          findutils

          # Go — builds/tests the RPC allowlist proxy used by the
          # forked-exploit task (docker/forked-exploit-compose/proxy-src/).
          # Unversioned attr tracks nixpkgs' current default (≥1.22 required
          # by go.mod). Dockerfile.proxy pins golang:1.22-alpine separately;
          # language is forward-compatible so the two don't have to match.
          go

          # Foundry — forge/anvil/cast. Used on the host by
          # scripts/warm_rpc_cache.py to pre-populate ~/.foundry/cache/rpc/
          # before baking it into Dockerfile.anvil. Dockerfile.foundry pins
          # its own foundry version for the in-image build.
          foundry
        ];

        shellHook = ''
          # NixOS exposes the NVIDIA user-mode driver (libcuda.so.1) here.
          if [ -d /run/opengl-driver/lib ]; then
            export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
            if [ -z "$TRITON_LIBCUDA_PATH" ] && [ -e /run/opengl-driver/lib/libcuda.so.1 ]; then
              export TRITON_LIBCUDA_PATH="/run/opengl-driver/lib"
            fi
          fi

          # libstdc++/libz for pip-installed torch/unsloth/triton wheels on NixOS.
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath wheelRuntimeLibs}:$LD_LIBRARY_PATH"

          if [ -z "$SSL_CERT_FILE" ]; then
            export SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
          fi

          # Host networking on Nix-based setups can require the host IP rather
          # than loopback when talking to the vLLM container.
          if [ -z "$NSL_VLLM_NETWORK_MODE" ]; then
            export NSL_VLLM_NETWORK_MODE=hostip
          fi

          echo "Nix dev shell ready (limiting-factor). Next: uv sync --all-extras"
        '';
      };
    };
}
