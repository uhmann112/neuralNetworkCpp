#pragma once
#include <vector>

class Neuron {
public:
    Neuron(int numInputs, size_t wheightStartIndex,
        std::vector<double>& weights);

    double processFW(const std::vector<double>& inputs);

    double sigmoid(double weightedSum);

    size_t weightStartIndex;

    double bias;

private:
    size_t numInputs;
    double weightedSum;
    std::vector<double>& weights;
};
