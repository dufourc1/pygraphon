import numpy as np
import matplotlib.pyplot as plt

# Define your scaling functions
def rho_log_over_n(n):
    return np.log(n) / n

def rho_one_over_sqrt_n(n):
    return 1 / np.sqrt(n)

def rho_constant(n):
    return 1 / n  # Example of another scaling function

# List of scaling functions
scaling_functions = {
    'log(n)/n': rho_log_over_n,
    '1/sqrt(n)': rho_one_over_sqrt_n,
    '1/n': rho_constant
    # Add more scaling functions as needed
}