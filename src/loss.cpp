#include "loss.h"
#include <iostream>
#include <cmath>
#include <cassert>

namespace penguin_net {

	double loss::MSE(double pred, double acc) {
		return (pred - acc) * (pred - acc);
	}

	double loss::MSEprime(double pred, double acc) {
		return 2 * (pred - acc);
	}

	// mean absolute value loss
	double loss::MAE(double pred, double acc) {
		return std::abs(pred - acc);
	}

	double loss::MAEprime(double pred, double acc) {
		return (pred - acc > 0 ? 1 : -1);
	}

	// (rows, 1)
	double loss::MSE(matrix &pred, matrix &acc) {
		assert(pred.shape() == acc.shape());
		return ((pred - acc).pow(2).sum() / pred.get_size());
	}

	penguin_net::matrix loss::MSEprime(matrix &pred, matrix &acc) {
		return (pred - acc)*(2)/(pred.get_size());
	}

	double loss::MAE(matrix &pred, matrix &acc) {
		assert(pred.shape() == acc.shape());
		return ((pred - acc).abs().sum() / pred.get_size());
	}

	penguin_net::matrix loss::MAEprime(matrix &pred, matrix &acc) {
		return (pred - acc).sign() / pred.get_size();
	}

	double loss::SEL(matrix &pred, matrix &acc) {
		return 0.5 * (pred-acc).pow(2).sum();
	}

	penguin_net::matrix loss::SELprime(matrix &pred, matrix &acc) {
		return pred - acc;
	}

	double loss::CCELoss(matrix &pred, matrix &acc) {
		assert(pred.shape() == acc.shape());
		return -(acc.hadamard(pred.log(exp(1.0))).sum());
	}

	// pred is softmaxxed.
	penguin_net::matrix loss::CCEprime(matrix &pred, matrix &acc) {
		return pred - acc;
	}

	double loss::BCELoss(matrix &pred, matrix &acc) {
		double loss = acc(0, 0) * std::log(pred(0, 0)) + acc(1, 0) * std::log(pred(1, 0));
		return -loss;
	}

	penguin_net::matrix loss::BCEprime(matrix &pred, matrix &acc) {
		return pred - acc;
	}
}
