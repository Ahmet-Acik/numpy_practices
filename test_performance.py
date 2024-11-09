import time
import numpy as np
import logging

def perform_statistical_analysis(data):
    # Dummy function to simulate statistical analysis
    return np.mean(data)

def test_performance():
    large_data = np.random.rand(1000000)
    start_time = time.time()
    perform_statistical_analysis(large_data)
    end_time = time.time()
    logging.info(f"Performance test took {end_time - start_time} seconds")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_performance()