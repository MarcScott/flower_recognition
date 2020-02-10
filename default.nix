with import <nixpkgs> {};
  
with pkgs.python37Packages;

stdenv.mkDerivation {
  name = "impurePythonEnv";


  buildInputs = [
    gcc6
    python3
    virtualenv
    matplotlib
    libgcc
    tkinter
    setuptools
    pkg-config
    setuptools
    ];
  src = null;
  # set SOURCE_DATE_EPOCH so that we can use python wheels
  shellHook = ''
  SOURCE_DATE_EPOCH=$(date +%s)
  export TMPDIR=$PWD/tmp
  virtualenv --no-setuptools venv
  export PATH=$PWD/venv/bin:$PATH
  pip install --no-cache-dir -r requirements.txt
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${gcc6.cc.lib};
  '';
}
