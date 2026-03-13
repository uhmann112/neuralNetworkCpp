#include "Neuron.h"
#include <cstdlib>
#include <ctime>
#include <cmath>


Neuron::Neuron(const std::vector<double>& inputs,size_t numInputs){
    this->numInputs = numInputs; //selbnsterklaerend
    // Initialize weights
    this->weights.resize(numInputs);
    for (size_t i = 0; i < numInputs; ++i) {
        this->weights[i] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
    }
    this->bias = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
    this->inputs = inputs;
}


void Neuron::processFw(const std::vector<double>& inputs,
                       const std::vector<double>& weights){
    this->weightedSum = 0.0;

    for (size_t i = 0; i < numInputs; ++i) {
        weightedSum += inputs[i] * weights[i];
    }
    weightedSum += bias;
    this->output=sigmoid(weightedSum);
}


double Neuron::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}
