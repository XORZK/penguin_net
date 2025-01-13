#ifndef LOSS_H
#define LOSS_H

#include "matrix.h"

namespace penguin_net::loss {
	double MSE(double a, double e); // mean square error
	double MSEprime(double a, double e);
	double MAE(double a, double e); // mean abs error
	double MAEprime(double a, double e);
	double MSE(matrix &a, matrix &e); // mean square error
	matrix MSEprime(matrix &a, matrix &e);
	double MAE(matrix &a, matrix &e); // mean abs error
	matrix MAEprime(matrix &a, matrix &e);
	double SEL(matrix &a, matrix &b); // square error loss
	matrix SELprime(matrix &a, matrix &b);
	double CCELoss(matrix &a, matrix &e); // categorical cross entropy loss
	matrix CCEprime(matrix &a, matrix &e);
	double BCELoss(matrix &a, matrix &e); // binary cross entropy loss
	matrix BCEprime(matrix &a, matrix &e);
};

#endif
