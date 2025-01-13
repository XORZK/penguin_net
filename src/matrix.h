#ifndef MATRIX_H
#define MATRIX_H

#pragma once
#include <vector>
#include <tuple>
#include <ostream>

// Very simple matrix structure.
namespace penguin_net {
    class matrix {
        private:
            int rows, cols;
            std::vector<double> values;
        public:
            matrix();
            matrix(int R, int C, bool random=false);
            matrix(int R, int C, double initial);
            matrix(std::tuple<int, int> shape);
            matrix(std::vector<double> copy, int R, int C);
            matrix(const matrix &mat);
            ~matrix();
            int get_size() const;
            int get_rows() const;
            int get_cols() const;
            std::vector<double> get_values() const;
            std::tuple<int, int> shape() const;
            double& operator()(int j);
            double operator()(int j) const;
            double& operator()(int j, int k);
            double operator()(int j, int k) const;
            matrix transpose() const;
            matrix flatten() const;
            matrix operator+(const matrix &m2) const;
            matrix operator-(const matrix &m2) const;
            matrix operator*(const matrix &m2) const;
            matrix operator*(const double scalar) const;
            matrix operator/(const double scalar) const;
            matrix& operator+=(const matrix &m2);
            matrix& operator-=(const matrix &m2);
            matrix& operator*=(const matrix &m2);
            matrix& operator*=(const double scalar);
            matrix& operator/=(const double scalar);
            bool operator==(const matrix &m2) const;
            bool operator!=(const matrix &m2) const;
            matrix norm() const;
            matrix hadamard(const matrix &m2) const;
            matrix pow(const double e) const;
            matrix log(const double b) const;
            matrix abs() const;
            matrix sign() const;
            double sum() const;
            void fill(const double value);
            void randomize(const double lower = 0, const double upper = 1);
            void int_random(const int lower = 0, const int upper = 5);
            void out() const;
    };
};



#endif
