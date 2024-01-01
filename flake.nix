{
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python311;
        pyenv = python.withPackages (ps: with ps; [
          numpy
          matplotlib
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            nodejs
            pyenv
            python311Packages.ipython
          ];
          DISPLAY = ":0";
        };
      });
}
