/**
 * @file value.cpp
 * @brief Implementation of the Value class for automatic differentiation
 *
 * This file contains the implementation of all Value class methods declared
 * in value.hpp.
 */

#include "micrograd/value.hpp"

#include <iostream>
#include <string>

// ======== VALUE CLASS =========
/**
 * @brief Constructor implementation
 * @param data Initial data value
 * @param label Optional label for debugging
 *
 * Uses member initializer list for efficiency
 */
Value::Value(double data, const std::string& label)
    : m_data(data), m_grad(0.0), m_label(label) {
    /*
     * m_grad is initialised to 0.0
     * - gradients start at zero and accumulate during backprop
     * - Safe default that won't interfere with gradient computation
     */
}

// ======= ACCESSORS =======

/**
 * @brief Get the data value
 * @return Current data value
 */
double Value::data() const {
    return m_data;
}

/**
 * @brief Get the gradient value
 * @return Current gradient value
 */
double Value::grad() const {
    return m_grad;
}

/**
 * @brief Get the label
 * @return const reference to label string
 *
 */
const std::string& Value::label() const {
    return m_label;
}

// ======= MUTATORS ========

/**
 * @brief Set the data value
 * @param data New data value
 */
void Value::set_data(double data) {
    m_data = data;
}

/**
 * @brief Set the gradient value
 * @param grad New gradient value
 */
void Value::set_grad(double grad) {
    m_grad = grad;
}

/**
 * @brief Add to the current gradient (accumulation)
 * @param grad_increment Value to add to current gradient
 */
void Value::add_to_grad(double grad_increment) {
    m_grad += grad_increment;
}

/**
 * @brief Reset gradient to zero
 */
void Value::zero_grad() {
    m_grad = 0.0;
}

/**
 * @brief Set the label
 * @param label New label string
 */
void Value::set_label(const std::string& label) {
    m_label = label;
}

// ======== UTILITY METHODS =======
/**
 * @brief Print Value to stdout for debugging
 */
void Value::print() const {
    std::cout << "Value(data=" << m_data << ", grad=" << m_grad;
    if (!m_label.empty()) {
        std::cout << ", label=\"" << m_label << "\"";
    }
    std::cout << ")" << std::endl;
}

/**
 * @brief Get string representation
 * @return String representation of the Value
 */
std::string Value::to_string() const {
    std::string result = "Value(" + std::to_string(m_data) + ")";
    if (!m_label.empty()) {
        result += "[" + m_label + "]";
    }
    return result;
}

// ======== FACTORY FUNCTIONS ========
/**
 * @brief Factory function to create Value with shared_ptr
 * @param data Initial data value
 * @param label Optional label
 * @return shared_ptr to newly created Value
 */
ValuePtr make_value(double data, const std::string& label) {
    return std::make_shared<Value>(data, label);
}
