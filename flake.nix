{
  description = "MMD2 emotion project — CUDA-enabled Python dev shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

        pythonEnv = pkgs.python312.withPackages (ps: with ps; [
          # Core scientific stack
          numpy
          packaging
          pandas
          scikit-learn
          matplotlib
          seaborn
          tqdm

          # Jupyter
          jupyter
          ipykernel
          ipywidgets
          nbclient
          nbformat
          notebook

          # ML / NLP
          torch              # CUDA-enabled via cudaSupport = true
          torchvision
          transformers
          accelerate
          bitsandbytes      # 8-bit AdamW (paged_adamw_8bit) for full fine-tuning within 24 GB
          datasets
          tokenizers
          safetensors
          sentencepiece

          # Explainability
          lime
        ]);

        # Runtime libraries PyTorch / NCCL / cuDNN dlopen.
        cudaLibs = with pkgs.cudaPackages; [
          cuda_cudart
          cuda_cupti
          cuda_nvrtc
          cuda_nvtx
          cudnn
          libcublas
          libcufft
          libcurand
          libcusolver
          libcusparse
          nccl
        ];

        systemLibs = with pkgs; [
          stdenv.cc.cc.lib
          zlib
        ];

        ldPath = pkgs.lib.makeLibraryPath (cudaLibs ++ systemLibs);
      in {
        devShells.default = pkgs.mkShell {
          packages = [
            pythonEnv
            pkgs.git
          ];

          shellHook = ''
            export CUDA_HOME=${pkgs.cudaPackages.cuda_nvcc}
            export LD_LIBRARY_PATH=${ldPath}:/run/opengl-driver/lib:$LD_LIBRARY_PATH

            # Keep HF caches inside the repo for easy cleanup.
            export HF_HOME="$PWD/.cache/huggingface"

            echo ">> Python:        $(python --version)"
            echo ">> Interpreter:   ${pythonEnv}/bin/python"
            echo ">> In VSCode pick that interpreter for the notebook kernel."
          '';
        };
      });
}
