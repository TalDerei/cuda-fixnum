#!/bin/bash
mkdir build
pushd build
  cmake -DMULTICORE=ON -DUSE_PT_COMPRESSION=OFF $EXTRA_CMAKE_ARGS_FOR_CI ..
  make -j12 main
popd
mv build/main .