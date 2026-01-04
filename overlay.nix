final: prev: {
  pythonPackagesExtensions = [
    (finalPythonPackages: prevPythonPackages: {
      torchvision = prevPythonPackages.torchvision.overrideAttrs (prevAttrs: {
        postPatch = ''
          nixLog "patching $PWD/setup.py to always use NVJPEG"
          substituteInPlace "$PWD/setup.py" \
            --replace-fail \
              'nvjpeg_found = CUDA_HOME is not None and (Path(CUDA_HOME) / "include/nvjpeg.h").exists()' \
              'nvjpeg_found = True'
        '';
        buildInputs = prevAttrs.buildInputs ++ [
          final.cudaPackages.libnvjpeg
        ];
      });
      # numba = prevPythonPackages.numba.overrideAttrs (prevAttrs: {

      # });
    })
  ];
}
