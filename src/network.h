#ifndef NETWORK_H
#define NETWORK_H

#pragma once
#include <iostream>
#include <vector>
#include "matrix.h"
#include "layer.h"
#include "loss.h"

namespace penguin_net {
	class network {
		private:
			std::vector<penguin_net::layer*> layers;
			std::string layer_type(int type = 0);
			bool has_input = false;
			int input_size, output_size;
			// loss function
			double (*cost)(matrix&, matrix&) = penguin_net::loss::MSE;
			matrix (*dcost)(matrix&, matrix&) = penguin_net::loss::MSEprime;
		public:
			int layer_count = 0;
			network();
			network(std::vector<std::size_t> lsize, std::vector<int> f = {});
			~network();
			void set_cost(const std::string f);
			void set_cost(double (*c)(matrix&, matrix&),
						  matrix (*dc)(matrix&, matrix&));
			void add(const int layer_size, const int layer_type = 0);
			void add(const int layer_size, std::string func);
			matrix forward(const matrix &in);
			void backward(matrix &in,
						  matrix &exp,
						  std::vector<matrix> &nabla_w,
						  std::vector<matrix> &deltas);
			void update_mini_batch(std::vector<std::pair<matrix, matrix>> b,
								   double lr = 0.01);
			void print() const;
			int num_layers() const;
	};
};

#endif
