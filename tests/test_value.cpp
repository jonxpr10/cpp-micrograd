/**
 * @file test_value.cpp
 * @brief Comprehensive unit tests for the Value class
 *
 */

#include "micrograd/value.hpp"  // Our Value class header
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>
#include <sstream>

// ======== TESTING FRAMEWORK ========

/**
 * Simple testing framework, TODO: use something like Google Test later
 */
class TestFramework {
private:
    int tests_run = 0;
    int tests_passed = 0;
    std::string current_test_name;

public:
    void start_test(const std::string& name) {
        current_test_name = name;
        std::cout << "Running: " << name << " ... ";
        tests_run++;
    }

    void assert_true(bool condition, const std::string& message = "") {
        if (condition) {
            std::cout << "✓ PASS" << std::endl;
            tests_passed++;
        } else {
            std::cout << "✗ FAIL";
            if (!message.empty()) {
                std::cout << " - " << message;
            }
            std::cout << std::endl;
        }
    }

    void assert_equal(double expected, double actual, double tolerance = 1e-9) {
        bool equal = std::abs(expected - actual) < tolerance;
        if (equal) {
            std::cout << "✓ PASS" << std::endl;
            tests_passed++;
        } else {
            std::cout << "✗ FAIL - Expected: " << expected
                      << ", Got: " << actual << std::endl;
        }
    }

    void assert_equal(const std::string& expected, const std::string& actual) {
        if (expected == actual) {
            std::cout << "✓ PASS" << std::endl;
            tests_passed++;
        } else {
            std::cout << "✗ FAIL - Expected: \"" << expected
                      << "\", Got: \"" << actual << "\"" << std::endl;
        }
    }
};

// =============================================================================
// UNIT TESTS
// =============================================================================

void test_value_construction(TestFramework& tf) {
    tf.start_test("Value Construction - Basic");
    Value v(3.14);
    tf.assert_equal(3.14, v.data());

    tf.start_test("Value Construction - With Label");
    Value labeled_v(2.71, "euler");
    tf.assert_equal(2.71, labeled_v.data());
    tf.assert_equal(std::string("euler"), labeled_v.label());

    tf.start_test("Value Construction - Default Gradient");
    tf.assert_equal(0.0, v.grad());

    tf.start_test("Value Construction - Empty Label Default");
    tf.assert_equal(std::string(""), v.label());
}

void test_value_getters_setters(TestFramework& tf) {
    tf.start_test("Data Getter/Setter");
    Value v(1.0);
    v.set_data(5.5);
    tf.assert_equal(5.5, v.data());

    tf.start_test("Gradient Getter/Setter");
    v.set_grad(2.3);
    tf.assert_equal(2.3, v.grad());

    tf.start_test("Label Getter/Setter");
    v.set_label("test_value");
    tf.assert_equal(std::string("test_value"), v.label());
}

void test_gradient_operations(TestFramework& tf) {
    tf.start_test("Gradient Accumulation");
    Value v(1.0);
    v.add_to_grad(0.5);
    v.add_to_grad(0.3);
    v.add_to_grad(0.2);
    tf.assert_equal(1.0, v.grad());

    tf.start_test("Zero Gradient");
    v.zero_grad();
    tf.assert_equal(0.0, v.grad());

    tf.start_test("Multiple Zero Grad Calls");
    v.add_to_grad(5.0);
    v.zero_grad();
    v.zero_grad();  // Should be safe to call multiple times
    tf.assert_equal(0.0, v.grad());
}

void test_edge_cases(TestFramework& tf) {
    tf.start_test("Zero Value");
    Value zero(0.0);
    tf.assert_equal(0.0, zero.data());

    tf.start_test("Negative Value");
    Value negative(-3.14);
    tf.assert_equal(-3.14, negative.data());

    tf.start_test("Large Value");
    Value large(1e10);
    tf.assert_equal(1e10, large.data());

    tf.start_test("Small Value");
    Value small(1e-10);
    tf.assert_equal(1e-10, small.data());

    tf.start_test("Empty String Label");
    Value empty_label(1.0, "");
    tf.assert_equal(std::string(""), empty_label.label());

    tf.start_test("Long String Label");
    std::string long_label(1000, 'a');  // 1000 'a' characters
    Value long_label_value(1.0, long_label);
    tf.assert_equal(long_label, long_label_value.label());
}

void test_factory_functions(TestFramework& tf) {
    tf.start_test("Factory Function - Basic");
    auto ptr = make_value(42.0);
    tf.assert_equal(42.0, ptr->data());

    tf.start_test("Factory Function - With Label");
    auto labeled_ptr = make_value(3.14, "pi");
    tf.assert_equal(3.14, labeled_ptr->data());
    tf.assert_equal(std::string("pi"), labeled_ptr->label());

    tf.start_test("Factory Function - Reference Counting");
    tf.assert_equal(1, labeled_ptr.use_count());

    tf.start_test("Factory Function - Shared Ownership");
    auto ptr1 = make_value(1.0);
    auto ptr2 = ptr1;
    tf.assert_equal(2, ptr1.use_count());
    tf.assert_equal(2, ptr2.use_count());
}

void test_memory_management(TestFramework& tf) {
    tf.start_test("Memory Management - Scope Test");
    ValuePtr outer_ptr;
    {
        auto inner_ptr = make_value(123.0, "scoped");
        outer_ptr = inner_ptr;
        tf.assert_equal(2, inner_ptr.use_count());
    }
    // inner_ptr is destroyed, but object should survive
    tf.assert_equal(1, outer_ptr.use_count());
    tf.assert_equal(123.0, outer_ptr->data());

    tf.start_test("Memory Management - Reset Test");
    auto ptr = make_value(456.0);
    tf.assert_true(ptr != nullptr, "Pointer should not be null");
    ptr.reset();
    tf.assert_true(ptr == nullptr, "Pointer should be null after reset");
}

void test_string_representation(TestFramework& tf) {
    tf.start_test("String Representation - No Label");
    Value v(2.5);
    std::string str = v.to_string();
    // Should contain the value, exact format may vary
    tf.assert_true(str.find("2.5") != std::string::npos,
                   "String should contain the value");

    tf.start_test("String Representation - With Label");
    Value labeled_v(1.5, "test");
    std::string labeled_str = labeled_v.to_string();
    tf.assert_true(labeled_str.find("1.5") != std::string::npos &&
                   labeled_str.find("test") != std::string::npos,
                   "String should contain both value and label");
}

void test_const_correctness(TestFramework& tf) {
    tf.start_test("Const Correctness - Const Object");
    const Value const_v(3.14, "pi");

    // These should compile and work with const objects
    double data = const_v.data();
    double grad = const_v.grad();
    std::string label = const_v.label();
    std::string str = const_v.to_string();

    tf.assert_equal(3.14, data);
    tf.assert_equal(0.0, grad);
    tf.assert_equal(std::string("pi"), label);
    tf.assert_true(!str.empty(), "String representation should not be empty");
}

void test_mathematical_operations(TestFramework& tf) {
    // --- Test addition ---
    tf.start_test("Addition Forward");
    auto a = make_value(2.0, "a");
    auto b = make_value(3.0, "b");
    auto c = a + b;
    tf.assert_equal(5.0, c->data());

    tf.start_test("Addition Backward");
    c->backward();
    tf.assert_equal(1.0, a->grad());
    tf.assert_equal(1.0, b->grad());

    // --- Test subtraction ---
    tf.start_test("Subtraction Forward");
    auto d = make_value(10.0, "d");
    auto e = make_value(4.0, "e");
    auto f = d - e;
    tf.assert_equal(6.0, f->data());

    tf.start_test("Subtraction Backward");
    f->backward();
    tf.assert_equal(1.0, d->grad());
    tf.assert_equal(-1.0, e->grad());

    // --- Test division ---
    tf.start_test("Division Forward");
    auto g = make_value(8.0, "g");
    auto h = make_value(2.0, "h");
    auto i = g / h;
    tf.assert_equal(4.0, i->data());

    tf.start_test("Division Backward");
    i->backward();
    tf.assert_equal(0.5, g->grad()); // 1/h
    tf.assert_equal(-2.0, h->grad()); // -g / h^2

    // --- Test mixed type operations ---
    tf.start_test("Mixed Type (value + double) Forward");
    auto j = make_value(5.0, "j");
    auto k = j + 10.0;
    tf.assert_equal(15.0, k->data());

    tf.start_test("Mixed Type (value + double) Backward");
    k->backward();
    tf.assert_equal(1.0, j->grad());

    tf.start_test("Mixed Type (double * value) Forward");
    auto m = make_value(3.0, "m");
    auto n = 4.0 * m;
    tf.assert_equal(12.0, n->data());

    tf.start_test("Mixed Type (double * value) Backward");
    n->backward();
    tf.assert_equal(4.0, m->grad());
}


// =============================================================================
// MAIN TEST RUNNER
// =============================================================================

int main() {
    std::cout << "Value Class Unit Tests" << std::endl;
    std::cout << "======================" << std::endl << std::endl;

    TestFramework tf;

    // Run all test suites
    std::cout << "--- Construction Tests ---" << std::endl;
    test_value_construction(tf);

    std::cout << "\n--- Getter/Setter Tests ---" << std::endl;
    test_value_getters_setters(tf);

    std::cout << "\n--- Gradient Operation Tests ---" << std::endl;
    test_gradient_operations(tf);

    std::cout << "\n--- Edge Case Tests ---" << std::endl;
    test_edge_cases(tf);

    std::cout << "\n--- Factory Function Tests ---" << std::endl;
    test_factory_functions(tf);

    std::cout << "\n--- Memory Management Tests ---" << std::endl;
    test_memory_management(tf);

    std::cout << "\n--- String Representation Tests ---" << std::endl;
    test_string_representation(tf);

    std::cout << "\n--- Const Correctness Tests ---" << std::endl;
    test_const_correctness(tf);

    std::cout << "\n--- Mathematical Operation & Backprop Tests ---" << std::endl;
    test_mathematical_operations(tf);

    return 0;
}
