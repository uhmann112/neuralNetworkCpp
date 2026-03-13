#include <iostream>
#include <vector>
#include <chrono>
#include "Layer.h"

int main() {
    srand(static_cast<unsigned>(time(nullptr)));

    // Mehr Neuronen = mehr Rechenlast
    Layer layer(3, 128);

    const int samples = 50000;
    std::vector<std::vector<double>> dataset(samples, std::vector<double>(3));

    // Chaotische, aber deterministische Inputs
    for (int i = 0; i < samples; i++) {
        double x = ((rand() % 2000) - 1000) / 1000.0;
        double y = ((rand() % 2000) - 1000) / 1000.0;
        double z = ((rand() % 2000) - 1000) / 1000.0;

        dataset[i] = { x, y, z };
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Reiner Forward‑Pass Benchmark
    for (int i = 0; i < samples; i++) {
        layer.forward(dataset[i]);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Zeit für " << samples << " Forward-Passes: "
              << ms << " ms" << std::endl;

    return 0;
}
