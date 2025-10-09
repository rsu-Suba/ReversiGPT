#!/bin/bash

PYBIND11_CMAKE_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")

mkdir -p build
cd build
cmake -DCMAKE_PREFIX_PATH="${PYBIND11_CMAKE_DIR}" ../cpp_reversi
cmake --build .
mv reversi_bitboard_cpp.* ..
mv reversi_mcts_cpp.* ..
cd ..
rm -rf build
echo "C++ module built successfully!"