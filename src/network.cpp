#include "network.h"

namespace penguin_net {
	std::string network::layer_type(int type) {
		std::string func = "";
		switch (type) {
			case (0): func = "sigmoid"; break;
			case (1): func = "tanh"; break;
			case (2): func = "relu"; break;
			case (3): func = "leaky_relu"; break; case (4): func = "softmax"; break;
			default: func = "sigmoid"; break;
		}

		return func;
	}

	network::network() {}

	// f: 0 -> sigmoid
	//    1 -> tanh
	//    2 -> relu
	//    3 -> leaky_relu
	//    4 -> softmax
	// lsize: 3 -> 3 -> 1
	network::network(std::vector<std::size_t> lsize, std::vector<int> f) {
		for (int j = 0; j < lsize.size(); j++) {
			std::size_t layer_size = lsize[j];
			int type = (j < f.size() ? f[j] :  0);
			add(layer_size, type);
		}
	}

	network::~network() {}

	void network::set_cost(const std::string f) {
		if (f == "MSE") {
			set_cost(loss::MSE, loss::MSEprime);
		} else if (f == "MAE") {
			set_cost(loss::MAE, loss::MAEprime);
		} else if (f == "SEL") {
			set_cost(loss::SEL, loss::SELprime);
		} else if (f == "CCE") {
			// output -> softmax
			layers[layers.size() - 1]->set_activation("softmax");
			set_cost(loss::CCELoss, loss::CCEprime);
		} else if (f == "BCE") {
			// output -> 1 : sigmoid
			assert(output_size == 1);
			layers[layers.size() - 1]->set_activation("sigmoid");
			set_cost(loss::BCELoss, loss::BCEprime);
		} else {
			set_cost(loss::MSE, loss::MSEprime);
		}
	}

	void network::set_cost(double (*c)(matrix&, matrix&),
										matrix (*dc)(matrix&, matrix&)) {
		cost = c;
		dcost = dc;
	}

	void network::add(const int layer_size, const int type) {
		std::string func = layer_type(type);
		add(layer_size, func);
	}

	void network::add(const int layer_size, std::string func) {
		if (layers.size() == 0 && !has_input) {
			input_size = layer_size;
			has_input = true;
			return;
		} else if (layers.size() == 0 && has_input) {
			layers.push_back(new layer(input_size, layer_size, func));;
			output_size = layer_size;
		} else {
			int out = (layers[layers.size() - 1])->output_size();
			layers.push_back(new layer(out, layer_size, func));
			output_size = layer_size;
		}

		++layer_count;
	}

	// forward pass
	matrix network::forward(const matrix &in) {
		matrix *c = new matrix(in);

		for (int j = 0; j < layer_count; ++j) {
			*c = layers[j]->forward(*c);
		}

		return *c;
	}

	void network::backward(matrix &in, matrix &exp, std::vector<matrix> &nabla_w, std::vector<matrix> &deltas) {
		matrix out = forward(in);
		layer *L = layers[layer_count-1];
		matrix delta_L = \
			dcost(out, exp).hadamard(L->acti_prime(L->z));

		deltas[layer_count-1] = delta_L;
		nabla_w[layer_count-1] = delta_L * layers[layer_count-2]->a.transpose();

		for (int j = layer_count-2; j >= 0; --j) {
			layer *c = layers[j];
			matrix delta = (layers[j+1]->weights.transpose() * deltas[j+1]).hadamard(c->acti_prime(c->z));
			deltas[j] = delta;
			nabla_w[j] = delta * (j == 0 ? in : layers[j-1]->a).transpose();
		}
	}

	void network::update_mini_batch(std::vector<std::pair<matrix, matrix>> b,
								    double lr) {
		std::vector<matrix> delta_w(layer_count), delta_b(layer_count);
		for (auto& [x, y] : b) {
			std::vector<matrix> nabla_w(layer_count), nabla_b(layer_count);
			backward(x, y, nabla_w, nabla_b);
			for (int j = 0; j < layer_count; j++) {
				if (delta_w[j].get_size() == 0) {
					delta_w[j] = nabla_w[j];
				} else {
					delta_w[j] += nabla_w[j];
				}

				if (delta_b[j].get_size() == 0) {
					delta_b[j] = nabla_b[j];
				} else {
					delta_b[j] += nabla_b[j];
				}
			}
		}

		double f = lr / b.size();

		for (int j = 0; j < layer_count; j++) {
			layers[j]->biases -= (delta_b[j] * f);
			layers[j]->weights -= (delta_w[j] * f);
		}
	}

	void network::print() const {
		for (int j = 0; j < layer_count; ++j) {
			layers[j]->print();
		}
	}
}
