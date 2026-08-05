#!/bin/sh
# -O2 matters a lot here: stb_image's PNG decoder runs ~2.5x faster than at -O0,
# and startup decodes a couple thousand frames.
set -e
clang++ -std=c++17 -O2 main.cpp glad/glad.o -o anim \
    -Iglad/include -I/opt/homebrew/include -L/opt/homebrew/lib \
    -lglfw -lavcodec -lavformat -lavutil -lswscale -lswresample \
    -framework OpenGL
