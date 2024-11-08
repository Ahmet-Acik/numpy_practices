import numpy as np

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
    


if __name__ == "__main__":
   pass