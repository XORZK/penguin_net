#include <iostream>
#include <ctime>
#include "network.h"

int main(void) {
    srand(time(NULL));

	std::vector<std::pair<penguin_net::matrix, penguin_net::matrix>> batch;
	penguin_net::matrix in(2, 1), out(1, 1);

	for (int j = 0; j < 10000; j++) {
		int a = rand() % 100, b = rand() % 100;
		int c = a + b;

		in(0, 0) = a;
		in(1, 0) = b;
		out(0, 0) = c;
		batch.push_back(std::make_pair(in, out));
	}

	std::vector<penguin_net::matrix> a(3), b(3);

	penguin_net::network n = penguin_net::network({2, 2, 1});
	//n.backward(in, out, a, b);
	//n.forward(in);
	n.update_mini_batch(batch);

	in(0, 0) = 3;
	in(1, 0) = 5;

	n.forward(in).out();
}
