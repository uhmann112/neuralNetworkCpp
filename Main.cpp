#include <iostream>
#include <vector>
#include <chrono>
#include "Layer.h"

int main() {
    srand(static_cast<unsigned>(time(nullptr)));

    // Layer(ANZAHL_DER_INPUTS,ANZAHL_DER NEURONEN)
    Layer layer(3, 3);

 


    int samples=200000;
    std::vector<double> dataset = {1.2,-0.2,2.3};
    std::vector<double> expected = {1.0,0.0,0.0};
    std::vector<double> softmaxVals;
    for (int i = 0; i < samples; i++) {
        std::vector output1 =layer.forward(dataset);
        softmaxVals=layer.softmax(output1);
        layer.learn(expected);
    }

    std::cout<<"expected: "<<std::endl;
    for(auto e:expected){
        std::cout<<e<<", ";
    }
    std::cout<<" "<<std::endl;
    for(auto n:softmaxVals){
        std::cout<<n<<", ";
    }
    std::cout<<" "<<std::endl;
    return 0;
}
