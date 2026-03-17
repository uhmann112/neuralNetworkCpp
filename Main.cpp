#include <iostream>
#include <vector>
#include <chrono>
#include "Layer.h"

int main() {
    srand(static_cast<unsigned>(time(nullptr)));

    // Layer(ANZAHL_DER_INPUTS,ANZAHL_DER NEURONEN)
    Layer layer(3, 3);

 


    int samples=20;
    std::vector<double> dataset = {1.2,-0.2,2.3};
    std::vector<double> softmaxVals;
    for (int i = 0; i < samples; i++) {
        std::vector output1 =layer.forward(dataset);
        softmaxVals=layer.softmax(output1);
    }
    for(auto n:softmaxVals){
        std::cout<<n<<", ";
    }
    std::cout<<" "<<std::endl;
    return 0;
}
