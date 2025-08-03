/**
 * @file value.hpp
 * @brief Value class declaration for automatic differentiation
 *
 * This header defines the core Value class that serves as a node in the
 * computational graph for automatic differentiation. Each Value stores
 * data, gradients, and provides the foundation for building neural networks.
 *
 */

#ifndef MICROGRAD_VALUE_HPP
#define MICROGRAD_VALUE_HPP

#include <functional> // For std::function
#include <memory>
#include <set> // For storing previous nodes
#include <string>
#include <vector>

/**
 * Forward declaration of Value class
 * This allows us to define ValuePtr before the full class definition
 */
class Value;

/**
 * Type alias for shared pointer to Value
 * Using shared_ptr enables:
 * - Automatic memory management
 * - Safe shared ownership in computational graphs
 * - Exception safety
 */
using ValuePtr = std::shared_ptr<Value>;

/**
 * @class Value
 * @brief Core computational graph node for automatic differentiation
 *
 * The Value class represents a single node in a computational graph.
 * It stores:
 * - data: The actual numerical value
 * - grad: Accumulated gradient (∂Loss/∂this_value)
 * - _backward: A function to compute the local gradient and propagate it
 * - _prev: A set of parent nodes in the graph
 * - _op: The operation that created this node
 * - label: Optional human-readable identifier
 *
 */
class Value : public std::enable_shared_from_this<Value> {
  private:
    double m_data;       ///< The actual numerical value
    double m_grad;       ///< Accumulated gradient ∂Loss/∂this_value
    std::string m_label; ///< Optional label for debugging

    // --- Graph-related members ---
    std::string m_op; ///< Operation that produced this value (e.g., "+", "*")
    std::set<ValuePtr> m_prev; ///< Set of parent nodes
    std::function<void()> m_backward_fn; ///< Function to run for backpropagation

  public:
    /**
     * @brief Construct a Value with data and optional label
     * @param data The numerical value to store
     * @param children The parent nodes of this value
     * @param op The operation that created this value
     * @param label Optional human-readable identifier
     */
    explicit Value(double data, const std::string &label = "");
    Value(double data, const std::set<ValuePtr> &children,
          const std::string &op = "", const std::string &label = "");

    ~Value() = default;

    /**
     * @brief Copy constructor
     * @param other Value to copy from
     *
     */
    Value(const Value& other) = default;

    /**
     * @brief Copy assignment operator
     * @param other Value to copy from
     * @return Reference to this object
     */
    Value& operator=(const Value& other) = default;

    /**
     * @brief Move constructor
     * @param other Value to move from
     */
    Value(Value&& other) noexcept = default;

    /**
     * @brief Move assignment operator
     * @param other Value to move from
     * @return Reference to this object
     */
    Value& operator=(Value&& other) noexcept = default;

    // ======= ACCESSORS =======
    /**
     * @brief Get the stored data value
     * @return The numerical data
     */
    double data() const;

    /**
     * @brief Get the accumulated gradient
     * @return The gradient value ∂Loss/∂this_value
     */
    double grad() const;
    const std::string &label() const;
    const std::set<ValuePtr>& prev() const;
    const std::string& op() const;

    /**
     * @brief Get the label
     * @return const reference to the label string
     */
    const std::string& label() const;

    // ======= MUTATORS =======

    /**
     * @brief Set the data value
     * @param data New data value to store
     */
    void set_data(double data);

    /**
     * @brief Set the gradient value
     * @param grad New gradient value
     */
    void set_grad(double grad);

    /**
     * @brief Add to the current gradient (accumulation)
     * @param grad_increment Value to add to current gradient
     */
    void add_to_grad(double grad_increment);

    /**
     * @brief Reset gradient to zero
     *
     * Essential for training - Gradients accumulate by default,
     * so they must be zeroed before each backward pass.
     * Otherwise gradients from previous iterations will interfere.
     */
    void zero_grad();

    /**
     * @brief Set the label
     * @param label New label string
     */
    void set_label(const std::string& label);

    // ======== BACKPROPAGATION ========
    /**
     * @brief Perform backpropagation from this Value
     *
     * Computes the gradient of all preceding nodes in the graph.
     * Assumes this Value is the final output of the graph (e.g. the loss).
     */
    void backward();

    // ======== UTILITY METHODS ========
    void print() const;

    /**
     * @brief Get string representation of the Value
     * @return String representation
     */
    std::string to_string() const;

    // ======== ACTIVATION FUNCTIONS ========
    friend ValuePtr tanh(const ValuePtr &v);
    friend ValuePtr exp(const ValuePtr &v);
    friend ValuePtr pow(const ValuePtr &base, double exp);
    friend ValuePtr operator+(const ValuePtr &lhs, const ValuePtr &rhs);
    friend ValuePtr operator*(const ValuePtr &lhs, const ValuePtr &rhs);
};

// ======== FACTORY FUNCTIONS =========
ValuePtr make_value(double data, const std::string &label = "");
ValuePtr make_value(double data, const std::set<ValuePtr> &children,
                    const std::string &op = "", const std::string &label = "");

// ======== OPERATOR OVERLOADS ========
ValuePtr operator-(const ValuePtr &lhs, const ValuePtr &rhs);
ValuePtr operator/(const ValuePtr &lhs, const ValuePtr &rhs);
ValuePtr operator-(const ValuePtr &v); // Negation

// Overloads for operations with doubles
ValuePtr operator+(const ValuePtr &lhs, double rhs);
ValuePtr operator+(double lhs, const ValuePtr &rhs);
ValuePtr operator*(const ValuePtr &lhs, double rhs);
ValuePtr operator*(double lhs, const ValuePtr &rhs);
ValuePtr operator-(const ValuePtr &lhs, double rhs);
ValuePtr operator-(double lhs, const ValuePtr &rhs);
ValuePtr operator/(const ValuePtr &lhs, double rhs);


#endif // MICROGRAD_VALUE_HPP