# adnn
**A neural network implementation with automatic compile-time backpropagation**

_Christian Hardmeier_

This is a C++ neural network toolkit that uses template metaprogramming to translate
a network specification into Eigen expression templates for both the forward and the
backward propagation pass at compile time. It contains a reimplementation of my
cross-lingual pronoun prediction network (nn6) whose original Matlab code is found
in [this repository](https://github.com/chardmeier/nn-pronouns).

I currently don't intend to extend or maintain this code any further as it just takes
too much time and other packages with good support and more features are readily available.

Roughly, the central parts of the code are organised as follows:

`nnet.h` - core data structures for network specifications and weights

`nnopt.h` - training algorithm

`netops.h` - basic network operations

`mlp.h` - multi-layer perceptron

`nn6.h` - the nn6 network, equivalent to the original Matlab version

`nn6-dev.h` - a development version of nn6 with some improvements over the original version

`vocmap.h` - vocabulary data structure

`3layer.cc` - a front-end for a 3-layer perceptron, for testing purposes

`nn6.cc` and `nn6-dev.cc` - front-ends for the two variants of nn6
