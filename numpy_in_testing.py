import numpy as np

# UI Testing
# Generate test data for form inputs
form_data = np.random.randint(0, 100, size=(10, 5))  # 10 rows, 5 columns

# Example: Validate image transformation
def validate_image_transformation(original_image, transformed_image):
    return np.array_equal(np.flipud(original_image), transformed_image)

original_image = np.array([[0, 1], [2, 3]])
transformed_image = np.array([[2, 3], [0, 1]])
assert validate_image_transformation(original_image, transformed_image), "Image transformation failed!"


# API Testing
import numpy as np
import requests

# Generate complex JSON payload
payload = {
    "data": np.random.rand(5, 3).tolist()  # Convert NumPy array to list for JSON serialization
}

response = requests.post("https://api.example.com/data", json=payload)
response_data = response.json()

# Validate numerical data in API response
expected_data = np.array(payload["data"])
actual_data = np.array(response_data["data"])
assert np.allclose(expected_data, actual_data), "API response data validation failed!"


# Database Testing
import numpy as np
import sqlite3

# Generate large dataset
large_dataset = np.random.rand(1000, 5)

# Insert data into database
conn = sqlite3.connect('test.db')
cursor = conn.cursor()
cursor.execute('CREATE TABLE IF NOT EXISTS test_table (col1 REAL, col2 REAL, col3 REAL, col4 REAL, col5 REAL)')
cursor.executemany('INSERT INTO test_table VALUES (?, ?, ?, ?, ?)', large_dataset.tolist())
conn.commit()

# Retrieve and validate data from database
cursor.execute('SELECT * FROM test_table')
db_data = np.array(cursor.fetchall())
assert np.allclose(large_dataset, db_data), "Database data validation failed!"

conn.close()


# Performance Testing
import numpy as np
import time

# Generate large dataset for performance testing
large_dataset = np.random.rand(1000000)

# Measure time taken for computation
start_time = time.time()

# Perform computation on large dataset
result = np.sum(large_dataset)

end_time = time.time()
execution_time = end_time - start_time

print(f"Computation result: {result}")
print(f"Execution time: {execution_time} seconds")


# Security Testing
import numpy as np
import hashlib

# Generate random data for hashing
data = np.random.bytes(1024)

# Compute hash of data
hash_result = hashlib.sha256(data).hexdigest()
print(f"Hash result: {hash_result}")


# Load Testing
import numpy as np
import requests
import threading

# Define load testing function
def send_request():
    response = requests.get("https://api.example.com")
    print(response.status_code)
    
# Generate multiple threads for load testing
num_threads = 10
threads = []

for _ in range(num_threads):
    thread = threading.Thread(target=send_request)
    threads.append(thread)
    thread.start()
    
for thread in threads:
    thread.join()
    
    
# Generate synthetic data for regression testing
X = np.random.rand(100, 1)  # 100 samples, 1 feature
y = 3 * X.squeeze() + 2 + np.random.randn(100) * 0.1  # Linear relation with noise

from sklearn.linear_model import LinearRegression

# Train a simple linear regression model
model = LinearRegression()
model.fit(X, y)

# Validate model predictions
predictions = model.predict(X)
mse = np.mean((predictions - y) ** 2)
print(f"Mean Squared Error: {mse}")


import numpy as np

# Generate high volume data
high_volume_data = np.random.rand(1000000, 10)  # 1 million rows, 10 columns

# Boundary testing
boundary_data = np.array([np.finfo(np.float64).max, np.finfo(np.float64).min])

print("High volume data and boundary data generated for stress testing.")

import numpy as np

# Generate data with potential duplicates
data_with_duplicates = np.random.randint(0, 10, size=(100, 5))

# Detect duplicates
unique_data, indices = np.unique(data_with_duplicates, axis=0, return_index=True)
duplicates = np.setdiff1d(np.arange(data_with_duplicates.shape[0]), indices)

print(f"Found {len(duplicates)} duplicate rows.")

import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic time series data
time_series_data = np.sin(np.linspace(0, 20, 100)) + np.random.randn(100) * 0.1

# Introduce anomalies
time_series_data[50:55] += 5

# Plot time series data
plt.plot(time_series_data)
plt.title("Synthetic Time Series Data with Anomalies")
plt.show()