{
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python311;
        pyenv = python.withPackages (ps: with ps; [
          matplotlib
          numpy
          scipy
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            pyenv
            python311Packages.ipython
          ];
          DISPLAY = ":0";
        };
      });
}
