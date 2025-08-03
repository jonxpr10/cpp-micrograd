/**
 * @file value.cpp
 * @brief Implementation of the Value class for automatic differentiation
 */

#include "micrograd/value.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
#include <string>
#include <vector>


// ======== VALUE CLASS CONSTRUCTORS =========
Value::Value(double data, const std::string& label)
    : m_data(data), m_grad(0.0), m_label(label), m_op(""), m_backward_fn([]() {}) {}

Value::Value(double data, const std::set<ValuePtr>& children, const std::string& op, const std::string& label)
    : m_data(data), m_grad(0.0), m_label(label), m_op(op), m_prev(children), m_backward_fn([]() {}) {}

// ======= ACCESSORS =======
double Value::data() const {
    return m_data;
}
double Value::grad() const {
    return m_grad;
}
const std::string& Value::label() const {
    return m_label;
}

// ======= MUTATORS ========
void Value::set_data(double data) {
    m_data = data;
}
void Value::set_grad(double grad) {
    m_grad = grad;
}
void Value::add_to_grad(double grad_increment) {
    m_grad += grad_increment;
}
void Value::zero_grad() {
    m_grad = 0.0;
}
void Value::set_label(const std::string& label) {
    m_label = label;
}


// ======== UTILITY METHODS =======
void Value::print() const {
    std::cout << "Value(data=" << m_data << ", grad=" << m_grad;
    if (!m_label.empty()) {
        std::cout << ", label=\"" << m_label << "\"";
    }
    std::cout << ")" << std::endl;
}

std::string Value::to_string() const {
    std::string result = "Value(" + std::to_string(m_data) + ")";
    if (!m_label.empty()) {
        result += "[" + m_label + "]";
    }
    return result;
}

// ======== FACTORY FUNCTIONS ========
ValuePtr make_value(double data, const std::string& label) {
    return std::make_shared<Value>(data, label);
}
ValuePtr make_value(double data, const std::set<ValuePtr>& children, const std::string& op, const std::string& label) {
    return std::make_shared<Value>(data, children, op, label);
}


// ======== OPERATOR OVERLOADS ========
ValuePtr operator+(const ValuePtr& lhs, const ValuePtr& rhs) {
    auto out = make_value(lhs->data() + rhs->data(), {lhs, rhs}, "+");
    out->m_backward_fn = [lhs, rhs, out]() {
        // Chain rule for addition: dL/dx = dL/dout * dout/dx = out.grad * 1.0
        lhs->add_to_grad(out->grad());
        rhs->add_to_grad(out->grad());
    };
    return out;
}

ValuePtr operator*(const ValuePtr& lhs, const ValuePtr& rhs) {
    auto out = make_value(lhs->data() * rhs->data(), {lhs, rhs}, "*");
    out->m_backward_fn = [lhs, rhs, out]() {
        // Chain rule for multiplication: dL/dx = dL/dout * dout/dx = out.grad * y
        lhs->add_to_grad(rhs->data() * out->grad());
        rhs->add_to_grad(lhs->data() * out->grad());
    };
    return out;
}

ValuePtr operator-(const ValuePtr& v) { // Unary negation
    return v * -1.0;
}

ValuePtr operator-(const ValuePtr& lhs, const ValuePtr& rhs) {
    // Subtraction is implemented as addition with a negated value
    return lhs + (-rhs);
}

ValuePtr operator/(const ValuePtr& lhs, const ValuePtr& rhs) {
    // Division is implemented as multiplication by the reciprocal
    return lhs * pow(rhs, -1.0);
}

// Overloads for double
ValuePtr operator+(const ValuePtr& lhs, double rhs_val) {
    return lhs + make_value(rhs_val);
}
ValuePtr operator+(double lhs_val, const ValuePtr& rhs) {
    return make_value(lhs_val) + rhs;
}
ValuePtr operator*(const ValuePtr& lhs, double rhs_val) {
    return lhs * make_value(rhs_val);
}
ValuePtr operator*(double lhs_val, const ValuePtr& rhs) {
    return make_value(lhs_val) * rhs;
}
ValuePtr operator/(const ValuePtr& lhs, double rhs_val) {
    return lhs / make_value(rhs_val);
}

