#include "Neuron.h"
#include <cstdlib>
#include <ctime>
#include <cmath>



Neuron::Neuron(int numInputs,
               size_t weightStartIndex,
               std::vector<double>& weights)
    : numInputs(numInputs),
      weightStartIndex(weightStartIndex),
      weights(weights)   // <-- MUSS hier stehen!
{
    this->bias = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
}


double Neuron::processFW(const std::vector<double>& inputs){
    this->weightedSum = 0.0;

    for (size_t i = 0; i < numInputs; ++i) {
        weightedSum += inputs[i] * weights[this->weightStartIndex+i];
    }
    weightedSum += bias;
    return sigmoid(weightedSum);
}


double Neuron::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}
