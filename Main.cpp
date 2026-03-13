#include <iostream>
#include "Neuron.h"

int main() {
    srand(time(nullptr));

    std::vector<double> inputs = {1.0, 2.0, 3.0};

    Neuron n(inputs);

    std::cout << "Running sanity check...\n";

    for (int i = 0; i < 100000; ++i){
        n.processFw(inputs, inputs);
    }
    std::cout << "Neuron output: " << n.output<< "\n";

    return 0;
}
