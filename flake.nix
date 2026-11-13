{
  # The patched Strata engine (strata/ = upstream v0.2.2 + patches/) for the Tesla P100, built with Nix.
  #
  #   nix build .#strata-engine --cores 6      # the engine alone
  #   nix build -o result                      # engine + server + tools (packages.default)
  #   result/bin/strata --version
  #   result/bin/strata-serve --engine strata --config configs/<config>.json --host 127.0.0.1 --port 8793
  #   result/bin/strata-tool iq_pack.py --gguf ... --out ...     # any script in strata/tools/
  #
  # Why not strata/flake.nix: that is upstream's (a dev shell on CUDA 13, which dropped Pascal), and strata/ must stay
  # exactly upstream + patches (tools/verify-source.sh).
  description = "Strata v0.2.2 + P100 patches: sm_60 engine on CUDA 12.6, server and tools";

  inputs = {
    # the nixpkgs strata/flake.lock pins upstream; it carries cudaPackages_12_6
    nixpkgs.url = "github:nixos/nixpkgs/6510408d147d8c5e6d76043e899e9251756f6a75";
    # the llama.cpp commit strata/CMakeLists.txt fetches (ggml for the build, gguf-py for the tools); as an input the
    # build needs no network (-DSTRATA_GGML_DIR)
    llama-cpp = {
      url = "github:ggml-org/llama.cpp/3cf03257f219afbe7334045ff7c6a06ac68c627d";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, llama-cpp }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;      # NVIDIA's CUDA toolkit
      };
      lib = pkgs.lib;

      cudaPackages = pkgs.cudaPackages_12_6;   # CUDA 13 cannot target sm_60
      stdenv = cudaPackages.backendStdenv;     # gcc 13, the newest host compiler nvcc 12.6 takes

      # The NVIDIA driver is the host's (Arch: /usr/lib), not a Nix package; it has to match the running kernel module.
      # Only the driver's own libraries are linked here, so no other host library mixes into the Nix closure.
      hostDriverLibDir = "/usr/lib";
      hostDriver = pkgs.runCommand "host-nvidia-driver-libs" { } ''
        mkdir -p $out/lib
        for l in libcuda.so libcuda.so.1 libnvidia-ml.so.1 libnvidia-ptxjitcompiler.so.1; do
          ln -s ${hostDriverLibDir}/$l $out/lib/$l
        done
      '';

      # strata/ as committed (a git flake sees tracked files only: no .venv, no build dirs)
      src = ./strata;

      strata-engine = stdenv.mkDerivation {
        pname = "strata-engine-p100";
        version = "0.2.2-p100";
        inherit src;

        nativeBuildInputs = [ pkgs.cmake pkgs.ninja pkgs.python3 cudaPackages.cuda_nvcc pkgs.makeWrapper ];
        buildInputs = with cudaPackages; [ cuda_cudart libcublas cuda_cccl ];

        cmakeFlags = [
          "-DCMAKE_BUILD_TYPE=Release"
          "-DSTRATA_ENABLE_CUDA=ON"
          "-DSTRATA_EXPERIMENTAL_SM60=ON"
          "-DCMAKE_CUDA_ARCHITECTURES=60-real"     # the P100 only; the M5000 (sm_52) is not a Strata card
          "-DSTRATA_BUILD_TESTS=OFF"
          # ggml-cpu for AVX2/FMA/F16C/BMI2 (Haswell) instead of -march=native, which Nix's compiler wrapper strips.
          # Strata's own AVX2/AVX-512 kernels are per-file and picked at run time either way.
          "-DSTRATA_PORTABLE=ON"
          "-DSTRATA_GGML_DIR=${llama-cpp}"
        ];

        buildPhase = ''
          runHook preBuild
          cmake --build . --target strata -j $NIX_BUILD_CORES
          runHook postBuild
        '';

        installPhase = ''
          runHook preInstall
          install -Dm755 strata $out/libexec/strata/strata
          makeWrapper $out/libexec/strata/strata $out/bin/strata \
            --prefix LD_LIBRARY_PATH : ${hostDriver}/lib
          runHook postInstall
        '';

        meta = {
          description = "Strata MoE engine, v0.2.2 + P100 patches, sm_60 / CUDA 12.6";
          license = lib.licenses.mit;
          platforms = [ system ];
          mainProgram = "strata";
        };
      };

      # strata/requirements.txt, from nixpkgs (its versions, not the file's pins)
      python = pkgs.python3.withPackages (ps: with ps; [
        numpy jinja2 regex pyyaml tqdm requests pillow psutil
      ]);

      strata-serve = pkgs.writeShellApplication {
        name = "strata-serve";
        runtimeInputs = [ python ];
        text = ''
          export PYTHONPATH=${src}''${PYTHONPATH:+:$PYTHONPATH}
          exec python -m serve.server "$@"
        '';
      };

      strata-tool = pkgs.writeShellApplication {
        name = "strata-tool";
        runtimeInputs = [ python ];
        text = ''
          if [ $# -lt 1 ]; then echo "usage: strata-tool <script in strata/tools, e.g. iq_pack.py> [args]" >&2; exit 2; fi
          tool=$1; shift
          export STRATA_GGUF_PY=${llama-cpp}/gguf-py
          exec python ${src}/tools/"$tool" "$@"
        '';
      };
    in
    {
      packages.${system} = {
        inherit strata-engine strata-serve strata-tool;
        default = pkgs.symlinkJoin {
          name = "strata-p100";
          paths = [ strata-engine strata-serve strata-tool ];
          # result/share/strata = the source the server runs from (data/expert-profile.bin etc. for the configs)
          postBuild = ''
            mkdir -p $out/share
            ln -s ${src} $out/share/strata
          '';
        };
      };

      devShells.${system}.default = (pkgs.mkShell.override { inherit stdenv; }) {
        packages = [ python pkgs.cmake pkgs.ninja cudaPackages.cuda_nvcc cudaPackages.cuda_cudart cudaPackages.libcublas ];
        shellHook = ''
          export STRATA_GGUF_PY=${llama-cpp}/gguf-py
          export LD_LIBRARY_PATH=${hostDriver}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
        '';
      };
    };
}
