#ifndef LAYER_H
#define LAYER_H

#pragma once
#include <iostream>
#include "matrix.h"
#include "activation.h"

namespace penguin_net {
    // weights, biases, gradients
    class layer {
        private:
            int in, out;
            bool softmax = false;
            double (*activation_function)(const double);
            double (*derivative)(const double);
        public:
            matrix weights, biases;
			matrix z, a;
            layer(std::size_t input, std::size_t output, std::string fname = "sigmoid");
            layer(std::size_t input, std::size_t output, double (*a_f)(const double x) = penguin_net::activation::sigmoid);
			void set_activation(std::string fname = "sigmoid");
            matrix forward(matrix in);
            int input_size() const;
            int output_size() const;
			double acti_prime(const double a) const;
			matrix acti_prime(const matrix &a) const;
			void print() const;
    };
};

#endif
