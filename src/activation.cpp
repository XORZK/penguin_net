#include "activation.h"

#include "matrix.h"
#include <iostream>
#include <cmath>

namespace penguin_net {

	double activation::sigmoid(const double x) {
		return 1.0 / (1.0 + exp(-x));
	}

	double activation::sigmoid_prime(const double x) {
		double s = sigmoid(x);
		return s * (1 - s);
	}

	double activation::tanh(const double x) {
		double a = exp(x), b = exp(-x);

		return (a-b)/(a+b);
	}

	double activation::tanh_prime(const double x) {
		double a = tanh(x);

		return 1 - (a*a);
	}

	double activation::relu(const double x) {
		return (x > 0 ? x : 0);
	}

	double activation::relu_prime(const double x) {
		return (x > 0 ? 1 : 0);
	}

	double activation::leaky_relu(const double x) {
		const double alpha = 0.01;
		return (x > (alpha * x) ? x : alpha*x);
	}

	double activation::leaky_relu_prime(const double x) {
		const double alpha = 0.01;
		return (x > (alpha * x) ? 1 : alpha);
	}

	std::vector<double> activation::softmax(const std::vector<double> &input) {
		double s = 0;
		std::size_t input_size = input.size();
		std::vector<double> out(input_size);

		for (int j = 0; j < input_size; j++) {
			s += exp(input[j]);
		}

		for (int j = 0; j < input_size; j++) {
			out[j] = exp(input[j]) / s;
		}

		return out;
	}

	std::vector<double> activation::softmax(const penguin_net::matrix &m) {
		matrix f = m.flatten();
		return penguin_net::activation::softmax(f.get_values());
	}

	penguin_net::matrix activation::activate(const penguin_net::matrix &m, double (*a_f)(const double x)) {
		int rows = m.get_rows(), cols = m.get_cols();
		matrix ret(rows, cols);

		for (int j = 0; j < rows; j++) {
			for (int k = 0; k < cols; k++) {
				ret(j, k) = a_f(m(j, k));
			}
		}

		return ret;
	}

	penguin_net::matrix activation::activate_prime(const penguin_net::matrix &m, double (*a_fp)(const double x)) {
		int rows = m.get_rows(), cols = m.get_cols();

		matrix ret(rows, cols);

		for (int j = 0; j < rows; j++) {
			for (int k = 0; k < cols; k++) {
				ret(j, k) = a_fp(m(j, k));
			}
		}

		return ret;
	}
}
