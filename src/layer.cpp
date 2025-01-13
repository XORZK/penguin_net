#include "layer.h"
#include <locale>
#include <map>

namespace penguin_net {
    layer::layer(std::size_t input, std::size_t output, double (*a_f)(const double x)) : in(input), out(output) {
        weights = matrix(output, input, true);
        biases = matrix(output, 1, true);
        activation_function = a_f;
    }

	// input x output = rows x cols
    layer::layer(std::size_t input, std::size_t output, std::string fname) : in(input), out(output) {
        weights = matrix(output, input, true);
        biases = matrix(output, 1, true);
		set_activation(fname);
	}

	void layer::set_activation(std::string fname) {
		std::map<std::string, double(*)(double)> a = {
			{"sigmoid", penguin_net::activation::sigmoid},
			{"tanh", penguin_net::activation::tanh},
			{"relu", penguin_net::activation::relu},
			{"leaky_relu", penguin_net::activation::leaky_relu},
		};

		std::map<std::string, double(*)(double)> a_p = {
			{"sigmoid", penguin_net::activation::sigmoid_prime},
			{"tanh", penguin_net::activation::tanh_prime},
			{"relu", penguin_net::activation::relu_prime},
			{"leaky_relu", penguin_net::activation::leaky_relu_prime},
		};


		bool matched = false;

		for (const auto& [k, v] : a) {
			if (k == fname) {
				activation_function = v;
				derivative = a_p[k];
				matched = true;
				break;
			}
		}

		if (!matched) {
			if (fname == "softmax") {
				softmax = true;
			} else {
				// default to relu
				activation_function = penguin_net::activation::relu;
				derivative = penguin_net::activation::relu_prime;
			}
		}
    }

	// in dim: (in, 1)
	penguin_net::matrix layer::forward(penguin_net::matrix inputs) {
		z = (weights * inputs) + biases;

		if (softmax) {
			std::vector<double> s = penguin_net::activation::softmax(z);
			return matrix(s, z.get_rows() * z.get_cols(), 1);
		}

		a = penguin_net::activation::activate(z, activation_function);
		return a;
	}

	int layer::input_size() const {
		return in;
	}

	int layer::output_size() const {
		return out;
	}

	double layer::acti_prime(const double b) const {
		return derivative(b);
	}

	matrix layer::acti_prime(const matrix &b) const {
		int rows = b.get_rows(), cols = b.get_cols();
		penguin_net::matrix r(rows, cols);

		for (int j = 0; j < rows; ++j) {
			for (int k = 0; k < cols; ++k) {
				r(j,k) = derivative(b(j,k));
			}
		}

		return r;
	}

	void layer::print() const {
		std::cout << "==============\nWEIGHTS:\n==============" << "\n";
		weights.out();
		std::cout << "==============\nBIASES:\n==============" << "\n";
		biases.out();
	}
}
