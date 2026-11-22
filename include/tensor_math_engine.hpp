#ifndef TENSOR_MATH_ENGINE_HPP
#define TENSOR_MATH_ENGINE_HPP
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
namespace amr {
inline double sigmoid_fn(double z) { return 1.0 / (1.0 + std::exp(-z)); }
inline int mcculloch_pitts(int x1, int x2, int theta = 2) { return (x1 + x2 >= theta) ? 1 : 0; }
}
#endif
