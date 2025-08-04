#include "micrograd/neuron.hpp"
#include <random> // For random weight initialization

// Helper function for random number generation
double random_uniform() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(-1.0, 1.0);
    return dis(gen);
}

Neuron::Neuron(int nin) {
    // Initialize weights with random values between -1 and 1
    for (int i = 0; i < nin; ++i) {
        m_w.push_back(make_value(random_uniform()));
    }
    // Initialize bias to 0 for simplicity (could also be random)
    m_b = make_value(0.0);
}

ValuePtr Neuron::operator()(const std::vector<ValuePtr> &x) {
    // Calculate the weighted sum + bias
    // act = sum(w*x) + b
    auto act = m_b; // Start with the bias
    for (size_t i = 0; i < m_w.size(); ++i) {
        act = act + m_w[i] * x[i];
    }
    // Apply the activation function
    return tanh(act);
}

std::vector<ValuePtr> Neuron::parameters() const {
    std::vector<ValuePtr> params = m_w;
    params.push_back(m_b);
    return params;
}