import numpy as np

'''
        Data Type and Data Structure in NumPy
    Data Type:

    ndarray: The primary data type in NumPy is the ndarray (N-dimensional array).
        Data Structure:

        ndarray: This is the main data structure used in NumPy for storing arrays of numbers. 
        
        Scalar: syntax: np.array(5) : A single number. For example, 5.         
        Vector: syntax: np.array([1, 2, 3]) :A 1D array of numbers. For example, [1, 2, 3].  
        Matrix: syntax: np.array([[1, 2], [3, 4]]) :A 2D array of numbers. For example, [[1, 2], [3, 4]].   
        Tensor: syntax: np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]) :An N-dimensional array of numbers. For example, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]. 
        
        
    Commonly Used Methods in NumPy
    Array Creation:

        np.array(): Create an array.
        np.zeros(): Create an array filled with zeros.
        np.ones(): Create an array filled with ones.
        np.arange(): Create an array with a range of values.
        np.linspace(): Create an array with linearly spaced values.
        np.logspace(): Create an array with logarithmically spaced values.
        
    
    Array Manipulation:

        np.reshape(): Change the shape of an array.
        np.resize(): Resize an array.
        np.ravel(): Flatten a multi-dimensional array into a 1D array.
        np.flatten(): Flatten a multi-dimensional array into a 1D array.
        np.transpose(): Transpose the dimensions of an array.
        np.concatenate(): Join two or more arrays.
        np.split(): Split an array into multiple sub-arrays.
        np.append(): Append values to the end of an array.
        np.insert(): Insert values into an array.
        np.delete(): Delete values from an array.
    
        

    Mathematical Operations:

        np.sum(): Sum of array elements.
        np.mean(): Mean of array elements.
        np.std(): Standard deviation of array elements.
        np.dot(): Dot product of two arrays.
        np.matmul(): Matrix product of two arrays.
        np.add(): Add two arrays.
        np.subtract(): Subtract two arrays.
        np.multiply(): Multiply two arrays.
        np.divide(): Divide two arrays.
        np.power(): Raise the elements of an array to a power.
        np.sqrt(): Compute the square root of an array.
        
    Random Number Generation:

        np.random.rand(): Generate an array of random numbers from a uniform distribution.
        np.random.randn(): Generate an array of random numbers from a normal distribution.
        np.random.randint(): Generate random integers.
        np.random.choice(): Generate random samples from a given 1D array.
        
    Indexing and Slicing:

        array[index]: Access elements using indices.
        array[start:stop:step]: Slice arrays.
        array[condition]: Filter arrays based on a condition. 
        

'''
# create an array from a list
arr = np.array([1, 2, 3, 4, 5])
print(arr)

# create a 2D array from a list of lists
arr_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(f"arr_2d {arr_2d}")

# create an array with zeros
arr_zeros = np.zeros(5, dtype=int, order='C')
print(arr_zeros)

# create an array with ones
arr_once =np.ones(5, dtype=int, order='C')
print(arr_once)

# create an array with a range of values
arr_range = np.arange(10, 20, 2) # start, stop, step
print(f"arr_range {arr_range}")

# create an array with linspace (linear space) 
arr_lins = np.linspace(10, 20, 2) # start, stop, num
print(f"arr_lins {arr_lins}")

# create an array with logspace (logarithmic space)
arr_logs = np.logspace(1, 10, num=5, base=10.0)
print(arr_logs)

# Reshape an array
arr_r = np.arange(8).reshape(2, 4)
print(f"Reshape arr_r {arr_r}")

# Flatten an array
arr_f = arr.flatten()
print(f"arr_f {arr_f}")

# Transpose an array 
arr_t = arr.transpose() 
print(f"arr_t {arr_t}") 

# Concatenate arrays
concatenated = np.concatenate((arr_2d, arr_r))
print(f"concatenated {concatenated}")

# Sum of array elements
sum_arr = np.sum(arr)
print(f"sum_arr {sum_arr}")

# Mean of array elements
mean_arr = np.mean(arr)
print(f"mean_arr {mean_arr}")

# Standard deviation of array elements
std_arr = np.std(arr)
print(f"std_arr {std_arr}")

# Dot product of two arrays
dot_product = np.dot(arr, arr)
print(f"dot_product {dot_product}")

# Matrix product of two arrays
matrix_product = np.matmul(arr_once, arr_t)
print(f"matrix_product {matrix_product}")   

# Generate random numbers
random_numbers = np.random.rand(5)
print(f"random_numbers: {random_numbers}")

def main():
    print("Hello World!")
    print(np.random.rand(5))
    
'''
Use Cases of NumPy in Software Testing

Data Generation for Testing:

    Random Data Generation: Use np.random methods to generate random data for testing algorithms and functions.
    Edge Cases: Create arrays with edge cases like very large or very small numbers, zeros, or negative values to test the robustness of the software.

Performance Testing:
    Large Data Sets: Generate large arrays to test the performance and scalability of software.
    Benchmarking: Use NumPy to create standardized data sets for benchmarking different implementations.

Validation and Verification:
    Expected Results: Use NumPy to compute expected results for comparison with actual results from the software.
    Statistical Analysis: Perform statistical analysis on test results to validate the correctness and reliability of the software.

Simulation and Modeling:
    Monte Carlo Simulations: Use NumPy for simulations that require random sampling and probabilistic modeling.
    Mathematical Modeling: Utilize NumPy for creating and testing mathematical models used in the software.
'''

# Generate random data for testing
random_data = np.random.rand(1000)

# Generate edge case data
edge_case_data = np.array([0, -1, 1e10, -1e10, np.inf, -np.inf, np.nan])

# Performance testing with large data sets
large_data_set = np.random.rand(1000000)

# Expected results for validation
def function_to_test(x):
    return x * 2

input_data = np.array([1, 2, 3, 4, 5])
expected_results = np.array([2, 4, 6, 8, 10])
actual_results = function_to_test(input_data)

# Validate results
assert np.array_equal(expected_results, actual_results), "Test failed!"

# Statistical analysis
mean = np.mean(random_data)
std_dev = np.std(random_data)

print(f"Mean: {mean}, Standard Deviation: {std_dev}")

# Monte Carlo simulation example
num_simulations = 10000
simulation_results = np.random.normal(loc=0, scale=1, size=num_simulations)
probability = np.mean(simulation_results > 1.96)

print(f"Probability of result > 1.96: {probability}")

if __name__ == "__main__":
   pass