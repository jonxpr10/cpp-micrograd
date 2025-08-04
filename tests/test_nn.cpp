/**
 * @file test_nn.cpp
 * @brief Unit tests for the neural network components (Neuron, Layer, MLP).
 */

#include "micrograd/neuron.hpp"
#include "micrograd/layer.hpp"
#include "micrograd/mlp.hpp"
#include "micrograd/value.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>

// ======== TESTING FRAMEWORK ========
// Re-using the simple testing framework from test_value.cpp

class TestFramework {
private:
    int tests_run = 0;
    int tests_passed = 0;
    std::string current_test_name;

public:
    ~TestFramework() {
        std::cout << "\n----------------------------------------\n";
        std::cout << "Test Summary: " << tests_passed << " / " << tests_run << " passed." << std::endl;
        std::cout << "----------------------------------------\n";
    }

    void start_test(const std::string& name) {
        current_test_name = name;
        std::cout << "Running: " << name << " ... ";
        tests_run++;
    }

    void pass() {
        std::cout << "✓ PASS" << std::endl;
        tests_passed++;
    }

    void fail(const std::string& message = "") {
        std::cout << "✗ FAIL";
        if (!message.empty()) {
            std::cout << " - " << message;
        }
        std::cout << std::endl;
    }

    void assert_true(bool condition, const std::string& message = "") {
        if (condition) {
            pass();
        } else {
            fail(message);
        }
    }

    void assert_equal(size_t expected, size_t actual, const std::string& message = "") {
        if (expected == actual) {
            pass();
        } else {
            std::string error_msg = "Expected: " + std::to_string(expected) + ", Got: " + std::to_string(actual);
            if (!message.empty()) {
                error_msg += " | " + message;
            }
            fail(error_msg);
        }
    }

    void assert_equal(double expected, double actual, double tolerance = 1e-9) {
        bool equal = std::abs(expected - actual) < tolerance;
        if (equal) {
            pass();
        } else {
            fail("Expected: " + std::to_string(expected) + ", Got: " + std::to_string(actual));
        }
    }
};

// =============================================================================
// NEURON TESTS
// =============================================================================
void test_neuron_suite(TestFramework& tf) {
    std::cout << "--- Neuron Class Tests ---" << std::endl;

    // Test Neuron Construction
    tf.start_test("Neuron Construction");
    Neuron n(3); // 3 inputs
    // A neuron has one weight per input, plus one bias.
    tf.assert_equal(static_cast<size_t>(4), n.parameters().size(), "Should have 3 weights + 1 bias");

    // Test Neuron Forward Pass
    tf.start_test("Neuron Forward Pass");
    Neuron n2(2); // 2 inputs
    auto x = std::vector<ValuePtr>{make_value(1.0), make_value(-2.0)};
    auto out = n2(x);
    // The output of tanh is always between -1 and 1.
    tf.assert_true(out->data() >= -1.0 && out->data() <= 1.0, "Output must be in range [-1, 1]");
}

// =============================================================================
// LAYER TESTS
// =============================================================================
void test_layer_suite(TestFramework& tf) {
    std::cout << "\n--- Layer Class Tests ---" << std::endl;

    // Test Layer Construction
    tf.start_test("Layer Construction");
    // A layer with 3 inputs and 4 outputs (4 neurons)
    Layer layer(3, 4);
    // Each of the 4 neurons has 3 weights and 1 bias.
    // Total parameters = 4 neurons * (3 weights + 1 bias) = 16
    tf.assert_equal(static_cast<size_t>(16), layer.parameters().size(), "Should be nout * (nin + 1) parameters");

    // Test Layer Forward Pass
    tf.start_test("Layer Forward Pass");
    Layer layer2(3, 5); // 3 inputs, 5 outputs
    auto x = std::vector<ValuePtr>{make_value(1.0), make_value(0.5), make_value(-1.0)};
    auto outs = layer2(x);
    // The output should be a vector of 5 values.
    tf.assert_equal(static_cast<size_t>(5), outs.size(), "Should have one output per neuron");
    // Check that each output is valid.
    bool all_valid = true;
    for (const auto& out : outs) {
        if (out->data() < -1.0 || out->data() > 1.0) {
            all_valid = false;
            break;
        }
    }
    tf.assert_true(all_valid, "All outputs must be in range [-1, 1]");
}

// =============================================================================
// MLP TESTS
// =============================================================================
void test_mlp_suite(TestFramework& tf) {
    std::cout << "\n--- MLP Class Tests ---" << std::endl;

    // Test MLP Construction
    tf.start_test("MLP Construction");
    // An MLP with 3 inputs, two hidden layers of 4 neurons each, and 1 output neuron.
    MLP mlp(3, {4, 4, 1});
    // Layer 1: 4 neurons * (3 inputs + 1 bias) = 16 params
    // Layer 2: 4 neurons * (4 inputs + 1 bias) = 20 params
    // Layer 3: 1 neuron  * (4 inputs + 1 bias) = 5 params
    // Total = 16 + 20 + 5 = 41
    tf.assert_equal(static_cast<size_t>(41), mlp.parameters().size());

    // Test MLP Forward Pass
    tf.start_test("MLP Forward Pass");
    MLP mlp2(3, {5, 2}); // 3 inputs -> 5 neurons -> 2 neurons
    auto x = std::vector<ValuePtr>{make_value(2.0), make_value(3.0), make_value(-1.0)};
    auto outs = mlp2(x);
    tf.assert_equal(static_cast<size_t>(2), outs.size(), "Final output should match last layer size");

    // Test MLP single output case
    tf.start_test("MLP Forward Pass (Single Output)");
    auto final_out = mlp(x); // Using the 3 -> 4 -> 4 -> 1 MLP
    tf.assert_equal(static_cast<size_t>(1), final_out.size(), "Final output should be size 1");
    tf.assert_true(final_out[0]->data() >= -1.0 && final_out[0]->data() <= 1.0, "Single output must be in range [-1, 1]");

    // Test Zero Grad
    tf.start_test("MLP Zero Grad");
    // 1. Create a network and do a forward/backward pass to get some gradients
    MLP mlp3(2, {2, 1});
    auto x_grad_test = std::vector<ValuePtr>{make_value(0.5), make_value(0.5)};
    auto final_val = mlp3(x_grad_test)[0];
    final_val->backward();

    // 2. Check that at least one parameter has a non-zero gradient
    auto params = mlp3.parameters();
    double grad_sum_before = 0.0;
    for(const auto& p : params) {
        grad_sum_before += p->grad();
    }
    // With random weights, it's astronomically unlikely for all gradients to be zero.
    // We check if the sum is not zero.
    tf.assert_true(std::abs(grad_sum_before) > 1e-12, "Gradients should be non-zero after backward()");

    // 3. Call zero_grad() and check that all gradients are now zero
    mlp3.zero_grad();
    double grad_sum_after = 0.0;
    for(const auto& p : params) {
        grad_sum_after += p->grad();
    }
    tf.assert_equal(0.0, grad_sum_after, 1e-12);
}


// =============================================================================
// MAIN TEST RUNNER
// =============================================================================
int main() {
    std::cout << "Neural Network Components Unit Tests" << std::endl;
    std::cout << "====================================" << std::endl << std::endl;

    TestFramework tf;

    test_neuron_suite(tf);
    test_layer_suite(tf);
    test_mlp_suite(tf);

    return 0; // The TestFramework destructor will print the summary
}
