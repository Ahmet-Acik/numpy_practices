import unittest
import numpy as np
from main import function_to_test, validate_results, perform_statistical_analysis, monte_carlo_simulation

class TestMainFunctions(unittest.TestCase):

    def test_function_to_test(self):
        input_data = np.array([1, 2, 3, 4, 5])
        expected_results = np.array([2, 4, 6, 8, 10])
        np.testing.assert_array_equal(function_to_test(input_data), expected_results)

    def test_validate_results(self):
        input_data = np.array([1, 2, 3, 4, 5])
        expected_results = np.array([2, 4, 6, 8, 10])
        validate_results(input_data, expected_results)

    def test_perform_statistical_analysis(self):
        random_data = np.random.rand(1000)
        mean, std_dev = perform_statistical_analysis(random_data)
        self.assertAlmostEqual(mean, 0.5, delta=0.1)
        self.assertAlmostEqual(std_dev, 0.29, delta=0.1)

    def test_monte_carlo_simulation(self):
        probability = monte_carlo_simulation(10000)
        self.assertGreater(probability, 0.02)
        self.assertLess(probability, 0.05)

    def test_edge_cases(self):
        # Test with empty array
        input_data = np.array([])
        expected_results = np.array([])
        np.testing.assert_array_equal(function_to_test(input_data), expected_results)

        # Test with large numbers
        input_data = np.array([1e10, 2e10, 3e10])
        expected_results = np.array([2e10, 4e10, 6e10])
        np.testing.assert_array_equal(function_to_test(input_data), expected_results)

        # Test with NaN and inf
        input_data = np.array([np.nan, np.inf, -np.inf])
        expected_results = np.array([np.nan, np.inf, -np.inf])
        np.testing.assert_array_equal(function_to_test(input_data), expected_results)

    def test_error_handling(self):
        # Test with invalid input
        with self.assertRaises(TypeError):
            function_to_test("invalid input")

if __name__ == '__main__':
    unittest.main()