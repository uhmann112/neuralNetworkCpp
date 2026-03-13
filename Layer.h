#pragma once
#include <vector>
#include "Neuron.h"

class Layer{
public:
	Layer(std::vector<double>& inputs, numNeurons);
	std::vector<Neuron> neuros;
	std::vector<double> forward(std::vector<double>& inputs);
	std::vector<double> layerWeights;
	int numNeurons;

}