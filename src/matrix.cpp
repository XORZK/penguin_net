#include "matrix.h"

#include <stdlib.h>
#include <cassert>
#include <iostream>
#include <numeric>
#include <cmath>

namespace penguin_net {
    matrix::matrix() : rows(0), cols(0) {}

    matrix::matrix(int R, int C, bool random) : rows(R), cols(C), values(R*C) {
        if (random) {
            randomize();
        }
    }

    matrix::matrix(int R, int C, double initial) : rows(R), cols(C), values(R*C) {
        fill(initial);
    }

    matrix::matrix(std::tuple<int, int> shape) : rows(std::get<0>(shape)), cols(std::get<1>(shape)), values(rows*cols) {
    }

    matrix::matrix(std::vector<double> copy, int R, int C) : rows(R), cols(C), values(R*C) {
        for (int j = 0; j < std::min(R*C, (int) copy.size()); j++) {
            values[j] = copy[j];
        }
    }

    matrix::matrix(const matrix &mat) : rows(mat.get_rows()), cols(mat.get_cols()), values(rows*cols) {
        for (int j = 0; j < values.size(); j++) {
            values[j] = mat(j);
        }
    }

    matrix::~matrix() {}

    int matrix::get_size() const {
        return rows * cols;
    }

    int matrix::get_rows() const {
        return rows;
    }

    int matrix::get_cols() const {
        return cols;
    }

    std::vector<double> matrix::get_values() const {
        return values;
    }

    std::tuple<int, int> matrix::shape() const {
        return { rows, cols };
    }

    double& matrix::operator()(int j) {
        assert(0 <= j && j < values.size());
        return values[j];
    }

    double matrix::operator()(int j) const {
        assert(0 <= j && j < values.size());
        return values[j];
    }

    // assumes matrix is 0 indexed.
    double& matrix::operator()(int j, int k) {
        assert(0 <= j && j < rows && 0 <= k && k < cols);

        return values[j * cols + k];
    }

    double matrix::operator()(int j, int k) const {
        assert(0 <= j && j < rows && 0 <= k && k < cols);
        return values[j * cols + k];
    }

    // Tranposing a square matrix represented by a 1-D vector:
    // m[(j, k)] --> m[(k, j)] 
    // 1 2 3      1 4 7 
    // 4 5 6 -->  2 5 8
    // 7 8 9      3 6 9
    // (0,1), (0, 2), (1, 2)
    // only have to swap (j, k) over/under (one or the other) the diagonal
    // Tranposing a rectangular matrix represented by a 1-D vector
    penguin_net::matrix matrix::transpose() const {
        matrix T(cols, rows);

        for (std::size_t j = 0; j < rows; j++) {
            for (std::size_t k = 0; k < cols; k++) {
                T(k, j) = values[j*cols+k];
            }
        }

        return T;
    }

    // Flatten matrix to (1, N) where N is the amount of cells
    // in the original matrix
    penguin_net::matrix matrix::flatten() const {
        matrix F(1, rows * cols);

        for (int j = 0; j < values.size(); j++) {
            F(0, j) = values[j];
        }

        return F;
    }

    penguin_net::matrix matrix::operator+(const matrix &m2) const {
        assert(this->rows == m2.get_rows() && this->cols == m2.get_cols());

        matrix S(this->rows, this->cols);

        for (int j = 0; j < this->rows; j++) {
            for (int k = 0; k < this->cols; k++) {
                S(j,k) = (*this)(j, k) + m2(j, k);
            }
        }

        return S;
    }

    penguin_net::matrix matrix::operator-(const matrix &m2) const {
        assert(this->rows == m2.get_rows() && this->cols == m2.get_cols());

        matrix D(this->rows, this->cols);

        for (int j = 0; j < this->rows; j++) {
            for (int k = 0; k < this->cols; k++) {
                D(j,k) = (*this)(j, k) - m2(j, k);
            }
        }

        return D;
    }

    // matmul
    // nxj * jxm --> nxm matrix
    penguin_net::matrix matrix::operator*(const matrix &m2) const {
        assert(this->cols == m2.get_rows());

        matrix P(this->rows, m2.get_cols(), 0.0);

        // dot.
        // jth row dot kth col
        // O(N^3)
        for (int j = 0; j < this->rows; j++) {
            for (int k = 0; k < m2.get_cols(); k++) {
                for (int l = 0; l < this->cols; l++) {
                    P(j, k) += (*this)(j, l) * m2(l, k);
                }
            }
        }

        return P;
    }

    penguin_net::matrix matrix::operator*(const double scalar) const {
        matrix S(this->rows, this->cols);

        for (int j = 0; j < this->rows; j++) {
            for (int k = 0; k < this->cols; k++) {
                S(j, k) = (*this)(j, k) * scalar;
            }
        }

        return S;
    }

    penguin_net::matrix matrix::operator/(const double scalar) const {
        assert(scalar != 0);
        matrix S(this->rows, this->cols);

        for (int j = 0; j < this->rows; j++) {
            for (int k = 0; k < this->cols; k++) {
                S(j, k) = (*this)(j, k)/scalar;
            }
        }

        return S;
    }

    penguin_net::matrix& matrix::operator+=(const matrix &m2) {
        assert(this->rows == m2.get_rows() && this->cols == m2.get_cols());

        for (int j = 0; j < this->rows; j++) {
            for (int k = 0; k < this->cols; k++) {
                this->values[j*this->cols + k] += m2(j, k);
            }
        }

        return (*this);
    }

    penguin_net::matrix& matrix::operator-=(const matrix &m2) {
        assert(this->rows == m2.get_rows() && this->cols == m2.get_cols());

        for (int j = 0; j < this->rows; j++) {
            for (int k = 0; k < this->cols; k++) {
                this->values[j*this->cols + k] -= m2(j, k);
            }
        }

        return (*this);
    }

    // matmul
    // resize to n x m.
    penguin_net::matrix& matrix::operator*=(const matrix &m2) {
        assert(this->cols == m2.get_rows());
        int c = m2.get_cols();
        std::vector<double> resized(this->rows * c);
        std::fill(resized.begin(), resized.end(), 0.0);

        for (int j = 0; j < this->rows; j++) {
            for (int k = 0; k < m2.get_cols(); k++) {
                for (int l = 0; l < this->cols; l++) {
                    resized[j * c + k] += (*this)(j, l) * m2(l, k);
                }
            }
        }

        this->cols = c;
        this->values = resized;

        return (*this);
    }

    penguin_net::matrix& matrix::operator*=(const double scalar) {
        for (int j = 0; j < this->values.size(); j++) {
            this->values[j] *= scalar; 
        }

        return (*this);
    }

    penguin_net::matrix& matrix::operator/=(const double scalar) {
        assert(scalar != 0);

        for (int j = 0; j < this->values.size(); j++) {
            this->values[j] /= scalar;
        }

        return (*this);
    }

    penguin_net::matrix matrix::norm() const {
        double sum = 0;
        for (int j = 0; j < this->values.size(); j++) 
            sum += this->values[j];
        assert(sum != 0);

        matrix N(this->rows, this->cols);

        for (int j = 0; j < this->values.size(); j++) {
            N(j) = this->values[j] / sum;
        }

        return N;
    }

    penguin_net::matrix matrix::hadamard(const matrix &m2) const {
        assert((m2.shape() == (*this).shape()));

        matrix H(this->rows, this->cols);

        for (int j = 0; j < this->rows; j++) {
            for (int k = 0; k < this->cols; k++) {
                H(j, k) = (*this)(j,k) * (m2)(j,k);
            }
        }

        return H;
    }

    penguin_net::matrix matrix::pow(const double e) const {
        matrix P(this->rows, this->cols);

        for (int j = 0; j < this->rows; j++) {
            for (int k = 0; k < this->cols; k++) {
                P(j, k) = std::pow((*this)(j,k), e);
            }
        }

        return P;
    }

    matrix matrix::log(const double b) const {
        assert(b > 1);
        matrix L(this->rows, this->cols);

        for (int j = 0; j < this->rows; j++) {
            for (int k = 0; k < this->cols; k++) {
                L(j, k) = log2((*this)(j,k)) / log2(b);
            }
        }

        return L;
    }

    matrix matrix::abs() const {
        matrix A(this->rows, this->cols);

        for (int j = 0; j < this->rows; j++) {
            for (int k = 0; k < this->cols; k++) {
                A(j, k) = std::abs((*this)(j, k));
            }
        }

        return A;
    }

    matrix matrix::sign() const {
        matrix S(this->rows, this->cols);

        for (int j = 0; j < this->rows; j++) {
            for (int k = 0; k < this->cols; k++) {
                double v = (*this)(j,k);
                S(j,k) = (v == 0 ? 0 : (v > 0 ? 1 : -1));
            }
        }

        return S;
    }

    double matrix::sum() const {
        double s = 0;

        for (double k : this->values) {
            s += k;
        }

        return s;
    } 

    void matrix::fill(const double value) {
        std::fill(this->values.begin(), this->values.end(), value);
    }

    // Fills the matrix with values R such that lower <= R < upper
    void matrix::randomize(const double lower, const double upper) {
        assert(upper > lower);
        for (int j = 0 ; j < this->rows * this->cols; j++) {
            double R = (upper-lower)*((double) rand() / (RAND_MAX)) + lower;
            this->values[j] = R;
        }
    }

    // lower <= R < upper
    void matrix::int_random(const int lower, const int upper) {
        assert(upper > lower);

        for (int j = 0; j < this->rows * this->cols; j++) {
            int R = (rand() % (upper - lower)) + lower;
            this->values[j] = R;
        }
    }

    void matrix::out() const {
        for (int j = 0; j < this->rows; j++) {
            for (int k = 0; k < this->cols; k++) {
                std::cout << (*this)(j, k) << " ";
            }
            std::cout << "\n";
        }
    }
};
