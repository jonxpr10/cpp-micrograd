#include "micrograd/layer.hpp"

Layer::Layer(int nin, int nout) {
    // Create nout neurons, each with nin inputs
    for (int i = 0; i < nout; ++i) {
        m_neurons.emplace_back(nin);
    }
}

std::vector<ValuePtr> Layer::operator()(const std::vector<ValuePtr> &x) {
    std::vector<ValuePtr> outs;
    outs.reserve(m_neurons.size()); // Pre-allocate memory for efficiency
    for (auto &neuron : m_neurons) {
        outs.push_back(neuron(x));
    }
    return outs;
}

std::vector<ValuePtr> Layer::parameters() const {
    std::vector<ValuePtr> params;
    for (const auto &neuron : m_neurons) {
        auto neuron_params = neuron.parameters();
        params.insert(params.end(), neuron_params.begin(), neuron_params.end());
    }
    return params;
}