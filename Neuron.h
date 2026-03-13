#pragma once
#include <vector>

class Neuron {
public:
    Neuron(const std::vector<double>& inputs,size_t numInputs);
    void processFw(const std::vector<double>& inputs,
                   const std::vector<double>& weights);

    double getWeightedSum() const { return weightedSum; }
    double sigmoid(double weightedSum);
    double output;
private:
    size_t numInputs;
    std::vector<double> inputs;
    std::vector<double> weights;
    double bias;
    double weightedSum;
};
