{
  inputs.lmaofit.url = "github:ercanar/pynixenv";
  inputs.lmaofit.inputs.nixpkgs.follows = "nixpkgs";

  outputs = { self, nixpkgs, flake-utils, lmaofit }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python311;

        flameprof = pkgs.python311Packages.buildPythonPackage rec {
          pname = "flameprof";
          version = "0.4";
          src = pkgs.fetchPypi {
            inherit pname version;
            hash = "sha256-28htQZDLu6Yk8eCkD0TZ25YTjidTTYPI70LUIIV4daM=";
          };
        };

        pyenv = python.withPackages (ps: with ps; [
          lmaofit.packages.${system}.default
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
            flameprof
          ];
        };
      });
}
