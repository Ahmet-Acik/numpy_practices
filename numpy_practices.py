import numpy as np

def generate_high_volume_data(rows=1000000, cols=10):
    """Generate high volume data for stress testing."""
    return np.random.rand(rows, cols)

def generate_boundary_data():
    """Generate boundary data for testing."""
    return np.array([np.finfo(np.float64).max, np.finfo(np.float64).min])

def detect_duplicates(data):
    """Detect duplicates in the given data."""
    unique_data, indices = np.unique(data, axis=0, return_index=True)
    duplicates = np.setdiff1d(np.arange(data.shape[0]), indices)
    return len(duplicates), duplicates

def generate_data_with_duplicates(rows=100, cols=5):
    """Generate data with potential duplicates."""
    return np.random.randint(0, 10, size=(rows, cols))

def main():
    # Generate high volume data
    high_volume_data = generate_high_volume_data()
    print("High volume data generated for stress testing.")

    # Generate boundary data
    boundary_data = generate_boundary_data()
    print("Boundary data generated for testing.")

    # Generate data with potential duplicates
    data_with_duplicates = generate_data_with_duplicates()
    num_duplicates, duplicates = detect_duplicates(data_with_duplicates)
    print(f"Found {num_duplicates} duplicate rows.")

if __name__ == "__main__":
    main()