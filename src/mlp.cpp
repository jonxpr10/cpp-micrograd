#include "micrograd/mlp.hpp"

MLP::MLP(int nin, const std::vector<int> &nouts) {
    // Create the sequence of layers
    int size = nin;
    for (int nout : nouts) {
        m_layers.emplace_back(size, nout);
        size = nout; // The input size for the next layer is the output size of this one
    }
}

std::vector<ValuePtr> MLP::operator()(std::vector<ValuePtr> x) {
    // Pass the input through each layer sequentially
    for (auto &layer : m_layers) {
        x = layer(x);
    }
    return x;
}

std::vector<ValuePtr> MLP::parameters() const {
    std::vector<ValuePtr> params;
    for (const auto &layer : m_layers) {
        auto layer_params = layer.parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}

void MLP::zero_grad() {
    for (const auto &p : parameters()) {
        p->zero_grad();
    }
}