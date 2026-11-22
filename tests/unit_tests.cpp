#include "tensor_math_engine.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
int main() {
    double w = 0.4, x = 1.5, t = 0.8, eps = 1e-5;
    auto loss = [&](double p) { double s = amr::sigmoid_fn(p * x); return 0.5 * (s - t) * (s - t); };
    double s = amr::sigmoid_fn(w * x);
    double ana = (s - t) * s * (1.0 - s) * x;
    double num = (loss(w + eps) - loss(w - eps)) / (2.0 * eps);
    assert(std::abs(ana - num) < 1e-7);
    assert(amr::mcculloch_pitts(1, 1, 2) == 1);
    std::cout << "[PASS] All Numerical Gradient & McCulloch-Pitts Unit Tests Verified!\n";
    return 0;
}
