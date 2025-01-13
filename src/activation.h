#ifndef ACTIVATION_H
#define ACTIVATION_H

#pragma once
#include "matrix.h"
#include <vector>

namespace penguin_net::activation {
    double sigmoid(const double x);
	double sigmoid_prime(const double x);
    double tanh(const double x);
	double tanh_prime(const double x);
    double relu(const double x);
	double relu_prime(const double x);
    double leaky_relu(const double x);
	double leaky_relu_prime(const double x);

    // todo: softmax
    std::vector<double> softmax(const std::vector<double> &in);
    std::vector<double> softmax(const penguin_net::matrix &m);

    // todo: pass function as parameter to generalize
    // activation functions on matrices
    // softmax does not work (operation with vectors)
    matrix activate(const penguin_net::matrix& m, double (*a_f)(const double x));
	matrix activate_prime(const penguin_net::matrix& m, double (*a_fp)(const double x));
};

#endif
