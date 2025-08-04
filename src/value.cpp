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
Value::Value(double data, const std::string &label)
    : m_data(data), m_grad(0.0), m_label(label), m_op(""), m_backward_fn([]() {}) {}

Value::Value(double data, const std::set<ValuePtr> &children, const std::string &op, const std::string &label)
    : m_data(data), m_grad(0.0), m_label(label), m_op(op), m_prev(children), m_backward_fn([]() {}) {}

// ======= ACCESSORS =======
double Value::data() const
{
    return m_data;
}
double Value::grad() const
{
    return m_grad;
}
const std::string &Value::label() const
{
    return m_label;
}

// ======= MUTATORS ========
void Value::set_data(double data)
{
    m_data = data;
}
void Value::set_grad(double grad)
{
    m_grad = grad;
}
void Value::add_to_grad(double grad_increment)
{
    m_grad += grad_increment;
}
void Value::zero_grad()
{
    m_grad = 0.0;
}
void Value::set_label(const std::string &label)
{
    m_label = label;
}

// ======== UTILITY METHODS =======
void Value::print() const
{
    std::cout << "Value(data=" << m_data << ", grad=" << m_grad;
    if (!m_label.empty())
    {
        std::cout << ", label=\"" << m_label << "\"";
    }
    std::cout << ")" << std::endl;
}

std::string Value::to_string() const
{
    std::string result = "Value(" + std::to_string(m_data) + ")";
    if (!m_label.empty())
    {
        result += "[" + m_label + "]";
    }
    return result;
}

// ======== FACTORY FUNCTIONS ========
ValuePtr make_value(double data, const std::string &label)
{
    return std::make_shared<Value>(data, label);
}
ValuePtr make_value(double data, const std::set<ValuePtr> &children, const std::string &op, const std::string &label)
{
    return std::make_shared<Value>(data, children, op, label);
}

// ======== OPERATOR OVERLOADS ========
ValuePtr operator+(const ValuePtr &lhs, const ValuePtr &rhs)
{
    auto out = make_value(lhs->data() + rhs->data(), {lhs, rhs}, "+");
    out->m_backward_fn = [lhs, rhs, out]()
    {
        // Chain rule for addition: dL/dx = dL/dout * dout/dx = out.grad * 1.0
        lhs->add_to_grad(out->grad());
        rhs->add_to_grad(out->grad());
    };
    return out;
}

ValuePtr operator*(const ValuePtr &lhs, const ValuePtr &rhs)
{
    auto out = make_value(lhs->data() * rhs->data(), {lhs, rhs}, "*");
    out->m_backward_fn = [lhs, rhs, out]()
    {
        // Chain rule for multiplication: dL/dx = dL/dout * dout/dx = out.grad * y
        lhs->add_to_grad(rhs->data() * out->grad());
        rhs->add_to_grad(lhs->data() * out->grad());
    };
    return out;
}

ValuePtr operator-(const ValuePtr &v)
{ // Unary negation
    return v * -1.0;
}

ValuePtr operator-(const ValuePtr &lhs, const ValuePtr &rhs)
{
    // Subtraction is implemented as addition with a negated value
    return lhs + (-rhs);
}

ValuePtr operator/(const ValuePtr &lhs, const ValuePtr &rhs)
{
    // Division is implemented as multiplication by the reciprocal
    return lhs * pow(rhs, -1.0);
}

// Overloads for double
ValuePtr operator+(const ValuePtr &lhs, double rhs_val)
{
    return lhs + make_value(rhs_val);
}
ValuePtr operator+(double lhs_val, const ValuePtr &rhs)
{
    return make_value(lhs_val) + rhs;
}
ValuePtr operator*(const ValuePtr &lhs, double rhs_val)
{
    return lhs * make_value(rhs_val);
}
ValuePtr operator*(double lhs_val, const ValuePtr &rhs)
{
    return make_value(lhs_val) * rhs;
}
ValuePtr operator/(const ValuePtr &lhs, double rhs_val)
{
    return lhs / make_value(rhs_val);
}

// ======== BACKPROPAGATION ========
void Value::backward()
{
    std::vector<ValuePtr> topo;
    std::set<ValuePtr> visited;

    // Recursively build a topologically sorted list of all nodes
    std::function<void(ValuePtr)> build_topo =
        [&](ValuePtr v)
    {
        if (visited.find(v) == visited.end())
        {
            visited.insert(v);
            for (const auto &child : v->m_prev)
            {
                build_topo(child);
            }
            topo.push_back(v);
        }
    };

    build_topo(shared_from_this());

    // The gradient of the final node with respect to itself is 1
    this->m_grad = 1.0;

    // Go backwards through the topologically sorted list and apply the chain rule
    std::reverse(topo.begin(), topo.end());
    for (const auto &v : topo)
    {
        if (v->m_backward_fn)
        {
            v->m_backward_fn();
        }
    }
}

// ======== ACTIVATION FUNCTIONS ========
ValuePtr tanh(const ValuePtr& v) {
   double t = std::tanh(v->data());
   auto out = make_value(t, {v}, "tanh");


   out->m_backward_fn = [v, t, out]() {
       // Chain rule for tanh: dL/dx = dL/dout * (1 - tanh(x)^2)
       v->add_to_grad((1 - t * t) * out->grad());
   };
   return out;
}


ValuePtr exp(const ValuePtr& v) {
   double e = std::exp(v->data());
   auto out = make_value(e, {v}, "exp");


   out->m_backward_fn = [v, e, out]() {
       // Chain rule for exp: dL/dx = dL/dout * exp(x)
       v->add_to_grad(e * out->grad());
   };
   return out;
}


ValuePtr pow(const ValuePtr& base, double exp_val) {
   double result = std::pow(base->data(), exp_val);
   auto out = make_value(result, {base}, "pow");


   out->m_backward_fn = [base, exp_val, out]() {
       // Chain rule for power: dL/dx = dL/dout * (n * x^(n-1))
       base->add_to_grad((exp_val * std::pow(base->data(), exp_val - 1)) * out->grad());
   };
   return out;
}

