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

#include <memory>
#include <string>

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
 * - label: Optional human-readable identifier
 *
 * Example usage:
 * @code
 * auto x = make_value(2.0, "input");
 * auto y = make_value(3.0, "weight");
 * // Future: auto z = x * y;  // Will create computational graph
 * @endcode
 */
class Value {
private:
    double m_data;          ///< The actual numerical value
    double m_grad;          ///< Accumulated gradient ∂Loss/∂this_value
    std::string m_label;    ///< Optional label for debugging

public:
    /**
     * @brief Construct a Value with data and optional label
     * @param data The numerical value to store
     * @param label Optional human-readable identifier
     *
     * The explicit keyword prevents implicit conversions to catch potential bugs where doubles might be accidentally
     * converted to Value objects.
     */
    explicit Value(double data, const std::string& label = "");

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

    // ======= UTILITY METHODS ========

    /**
     * @brief Print the Value to stdout
     *
     * Useful for debugging and development.
     */
    void print() const;

    /**
     * @brief Get string representation of the Value
     * @return String representation
     */
    std::string to_string() const;
};

// ======== FACTORY FUNCTIONS =========

/**
 * @brief Create a Value wrapped in shared_ptr
 * @param data The numerical value
 * @param label Optional label
 * @return shared_ptr to the created Value
 *
 */
ValuePtr make_value(double data, const std::string& label = "");

#endif // MICROGRAD_VALUE_HPP
