with import <nixpkgs> {};
  
with pkgs.python37Packages;

stdenv.mkDerivation {
  name = "impurePythonEnv";


  buildInputs = [
    # these packages are required for virtualenv and pip to work:
    #
    gcc6
    python3
    python37Packages.virtualenv
    python37Packages.matplotlib
    # the following packages are related to the dependencies of your python
    # project.
    # In this particular example the python modules listed in the
    # requirements.tx require the following packages to be installed locally
    # in order to compile any binary extensions they may require.
    #
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
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${gcc6.cc.lib}';
  '';
}
