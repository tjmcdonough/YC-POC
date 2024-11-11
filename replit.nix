{pkgs}: {
  deps = [
    pkgs.rustc
    pkgs.libiconv
    pkgs.cargo
    pkgs.glibcLocales
    pkgs.bash
    pkgs.zlib
    pkgs.tk
    pkgs.tcl
    pkgs.libxcrypt
    pkgs.libwebp
    pkgs.libtiff
    pkgs.libjpeg
    pkgs.libimagequant
    pkgs.lcms2
    pkgs.xcbuild
    pkgs.swig
    pkgs.openjpeg
    pkgs.mupdf
    pkgs.libjpeg_turbo
    pkgs.jbig2dec
    pkgs.harfbuzz
    pkgs.gumbo
    pkgs.freetype
    pkgs.postgresql
  ];
}
