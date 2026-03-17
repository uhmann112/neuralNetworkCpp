#include "Layer.h"
#include "Neuron.h"
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <vector>



Layer::Layer(int inputSize,int numNeurons){
	this->inputSize=inputSize;
	this->numNeurons =numNeurons;
	this->deltas.resize(numNeurons);

	fillWeights();
	initializeNeurons();


}

void Layer::fillWeights(){
	size_t num = this->numNeurons*this->inputSize;
	this->layerWeights.resize(num);

	for (size_t i = 0; i < num; ++i) {
        this->layerWeights[i] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
    }
}

void Layer::initializeNeurons(){
	size_t index=0;
	for (size_t i = 0; i < this->numNeurons; ++i) {
		neurons.emplace_back(inputSize, index, layerWeights);
        index+=this->inputSize;
    }
}

std::vector<double> Layer::forward(const std::vector<double>& inputs) {
	layerInputs=inputs;
    output.clear();

    for (int i = 0; i < numNeurons; ++i) {
        output.push_back(neurons[i].processFW(inputs));
    }

    return output;
}

std::vector<double> Layer::softmax(const std::vector<double>& inputs){
	double sum=0;
	std::vector<double> softOut(inputs.size());
	for (int i = 0; i < inputs.size(); ++i){
		double sq= inputs[i]*inputs[i];
		softOut[i]=sq;
		sum+=sq;
	}
	for (int i = 0; i < inputs.size(); ++i){
		softOut[i]=softOut[i]/sum;
	}
	return softOut;
}

void Layer::learn(std::vector<double>& expected){
	//itterieren ueber alle outputs aus FORWARD und packe die diff zwischen OUT und EXPECTED in DELTAS
	for (int i = 0; i < output.size(); ++i){
		deltas[i]=output[i]-expected[i];
	}
	

	//jedes NEURON hat einen stIndex, von diesem index aus itterieren wir ueber alle inputs und weights
	//(weights hat die groesse inputs*neurons) deltas ist so grpos wie neurons deswegen iterieren wir ueber i!
	int index=0;
	for (int i = 0; i < numNeurons; ++i){
		index=neurons[i].weightStartIndex;
		for (int j = 0; j < inputSize; ++j){
			layerWeights[index+j]-= 0.01*layerInputs[j]*deltas[i];
		}
		neurons[i].bias-=0.01*deltas[i];
	}

}






