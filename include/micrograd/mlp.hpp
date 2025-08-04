#ifndef MICROGRAD_MLP_HPP
#define MICROGRAD_MLP_HPP

#include "layer.hpp"
#include <vector>

/**
 * @class MLP
 * @brief A Multi-Layer Perceptron (the full neural network).
 */
class MLP {
  public:
    /**
     * @brief Construct the MLP.
     * @param nin The number of inputs to the network.
     * @param nouts A vector of integers specifying the size of each output layer.
     */
    MLP(int nin, const std::vector<int> &nouts);

    /**
     * @brief Perform the full forward pass through all layers.
     * @param x The initial input vector.
     * @return A vector of ValuePtrs from the final layer.
     */
    std::vector<ValuePtr> operator()(std::vector<ValuePtr> x);

    /**
     * @brief Get all parameters from all layers in the network.
     * @return A single vector containing all network parameters.
     */
    std::vector<ValuePtr> parameters() const;

    /**
     * @brief Zeros the gradients of all parameters in the network.
     * Essential to call before each backpropagation pass in a training loop.
     */
    void zero_grad();

  private:
    std::vector<Layer> m_layers; ///< The layers of the network
};

#endif // MICROGRAD_MLP_HPP