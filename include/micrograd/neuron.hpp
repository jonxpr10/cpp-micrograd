#ifndef MICROGRAD_NEURON_HPP
#define MICROGRAD_NEURON_HPP

#include "value.hpp"
#include <vector>

/**
 * @class Neuron
 * @brief A single neuron in a neural network layer.
 *
 * A neuron computes a weighted sum of its inputs, adds a bias,
 * and then applies an activation function (tanh).
 */
class Neuron {
  public:
    /**
     * @brief Construct a Neuron.
     * @param nin The number of inputs to the neuron.
     */
    explicit Neuron(int nin);

    /**
     * @brief Perform the forward pass for the neuron.
     * @param x A vector of ValuePtrs representing the inputs.
     * @return The single output ValuePtr from the neuron.
     */
    ValuePtr operator()(const std::vector<ValuePtr> &x);

    /**
     * @brief Get all parameters (weights and bias) of the neuron.
     * @return A vector of ValuePtrs containing the neuron's parameters.
     */
    std::vector<ValuePtr> parameters() const;

  private:
    std::vector<ValuePtr> m_w; ///< The weights of the neuron
    ValuePtr m_b;              ///< The bias of the neuron
};

#endif // MICROGRAD_NEURON_HPP