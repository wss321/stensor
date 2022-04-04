# STensor
STensor is a tiny C++ library for pytorch-like deep learning framework.

## Dependency
1. cmake
2. glog
3. gflag
4. gtest
5. ProtoBuffer
6. CUDA
7. OpenBlas
8. Boost/thread
## Component
### common
### math
### tensor
### proto
### memory_op
### random

# Compile
```bash
mkdir build&cd build
cmake ..
make -j 8
```
# MNIST Demo
## get dataset
```bash
cd data/mnist
sh ./get_mnist.sh
cd ../..
```
## run
```bash
cd <builddir>/exc
./mnist_train
```
