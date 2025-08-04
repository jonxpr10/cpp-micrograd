#ifndef MICROGRAD_LAYER_HPP
#define MICROGRAD_LAYER_HPP

#include "neuron.hpp"
#include <vector>

/**
 * @class Layer
 * @brief A layer of neurons in a neural network.
 */
class Layer {
  public:
    /**
     * @brief Construct a Layer of neurons.
     * @param nin The number of inputs for each neuron in the layer.
     * @param nout The number of neurons in the layer (i.e., the output size).
     */
    Layer(int nin, int nout);

    /**
     * @brief Perform the forward pass for the entire layer.
     * @param x A vector of ValuePtrs representing the inputs.
     * @return A vector of ValuePtrs representing the outputs of all neurons.
     */
    std::vector<ValuePtr> operator()(const std::vector<ValuePtr> &x);

    /**
     * @brief Get all parameters from all neurons in the layer.
     * @return A single vector containing all parameters.
     */
    std::vector<ValuePtr> parameters() const;

  private:
    std::vector<Neuron> m_neurons; ///< The neurons in this layer
};

#endif // MICROGRAD_LAYER_HPP