import unittest
import numpy as np
import logging
from numpy_practices import generate_high_volume_data, generate_boundary_data, detect_duplicates, generate_data_with_duplicates

# Configure logging
logging.basicConfig(level=logging.INFO)

class TestNumpyPractices(unittest.TestCase):

    def test_generate_high_volume_data(self):
        logging.info("Testing generate_high_volume_data")
        data = generate_high_volume_data()
        logging.info(f"Generated data shape: {data.shape}")
        self.assertEqual(data.shape, (1000000, 10))

    def test_generate_boundary_data(self):
        logging.info("Testing generate_boundary_data")
        data = generate_boundary_data()
        logging.info(f"Generated boundary data: {data}")
        self.assertEqual(data.shape, (2,))
        self.assertEqual(data[0], np.finfo(np.float64).max)
        self.assertEqual(data[1], np.finfo(np.float64).min)

    def test_generate_data_with_duplicates(self):
        logging.info("Testing generate_data_with_duplicates")
        data = generate_data_with_duplicates()
        logging.info(f"Generated data shape: {data.shape}")
        self.assertEqual(data.shape, (100, 5))

    def test_detect_duplicates(self):
        logging.info("Testing detect_duplicates")
        data = generate_data_with_duplicates()
        num_duplicates, duplicates = detect_duplicates(data)
        logging.info(f"Number of duplicates: {num_duplicates}")
        logging.info(f"Duplicate indices: {duplicates}")
        self.assertIsInstance(num_duplicates, int)
        self.assertIsInstance(duplicates, np.ndarray)

if __name__ == '__main__':
    unittest.main()