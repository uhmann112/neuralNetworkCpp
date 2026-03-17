#include "Neuron.h"
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <vector>


class Layer {
public:
    Layer(int inputSize, int numNeurons);

    std::vector<double> forward(const std::vector<double>& inputs);
    std::vector<double> softmax(const std::vector<double>& inputs);
    void fillWeights();
    void initializeNeurons();

    int numNeurons;
    int inputSize;

    std::vector<double> layerWeights;
    std::vector<Neuron> neurons;
    std::vector<double> output;
};