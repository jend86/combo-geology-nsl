{
  description = "negative-space-learning dev environment";

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
      llamaCppCuda = pkgs.llama-cpp.override {
        cudaSupport = true;
      };
    in {
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

          # Useful for local GGUF tooling and conversion workflows.
          llamaCppCuda
        ];

        shellHook = ''
          # WSL CUDA driver (libcuda.so) lives outside the Nix store.
          if [ -d /usr/lib/wsl/lib ]; then
            export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
          fi

          # Triton can fail to resolve libcuda via ldconfig on Nix systems.
          if [ -z "$TRITON_LIBCUDA_PATH" ]; then
            for d in /usr/lib/wsl/lib /usr/lib64-nvidia /run/opengl-driver/lib /run/opengl-driver-32/lib; do
              if [ -e "$d/libcuda.so.1" ]; then
                export TRITON_LIBCUDA_PATH="$d"
                break
              fi
            done
          fi

          if [ -d /usr/lib64-nvidia ]; then
            export LD_LIBRARY_PATH="/usr/lib64-nvidia:$LD_LIBRARY_PATH"
          fi

          if [ -d /usr/local/cuda/lib64 ]; then
            export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
          fi

          if [ -z "$SSL_CERT_FILE" ]; then
            export SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
          fi

          # Host networking on Nix-based setups can require the host IP rather
          # than loopback when talking to the vLLM container.
          if [ -z "$NSL_VLLM_NETWORK_MODE" ]; then
            export NSL_VLLM_NETWORK_MODE=hostip
          fi

          # Convenience symlinks for llama.cpp binaries in-repo.
          LLAMA_CPP_DIR="$PWD/.llama"
          mkdir -p "$LLAMA_CPP_DIR"
          for bin in llama-quantize llama-cli llama-server llama-gguf-split; do
            if [ -e "${llamaCppCuda}/bin/$bin" ]; then
              ln -sf "${llamaCppCuda}/bin/$bin" "$LLAMA_CPP_DIR/$bin"
            fi
          done

          echo "Nix dev shell ready. Next: uv sync --all-extras"
        '';
      };
    };
}
