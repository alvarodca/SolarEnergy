import os
import csv
import pandas as pd
import numpy as np
import random 

# Constants

K_B_J = 1.380649 * 10^23 # Boltzman Constant J/K
N_T = 10^12 # Defect density
N_DOP = 5.1 * 10^15 # Doping density
DOPING_TYPE = "p"
NOISE = 0.01
delta_n_no_noise = 10^17

# Masses
M_E  = 9.10938356e-31 # Electron effective mass

# For Si
mass_n = 1.08* M_E # Effective mass of electrons
mass_p = 0.81 * M_E # Effective mass of holes

def thermal_velocity(T,eff_m):
    "Computes thermal velocity according to Green's model"
    return np.sqrt((3* K_B_J*T)/eff_m)

# Generating dataset
def generate_dataset(N):
    """Generates a dataset of N rows"""
    e_t = np.random.uniform(-0.55,0.55, N) # Defect Energy Level
    sigma_p = 10**np.random.uniform(-17,-13,N) # Capture cross sections p
    sigma_n = 10**np.random.uniform(-17,-13,N) # Capture cross section n
    k = sigma_n/sigma_p # Capture cross section ratio
    delta_n = 10**np.random.uniform(10^13, 10^17, N) # Excess carrier concentration
    temperature = np.random.uniform(200,400, N)
    alpha = np.random.uniform(1.1,4, N) # Power law coefficient
    e_v = np.random.uniform(0.01,0.25) # Activation Energy
    
    # Thermal velocities
    v_n = thermal_velocity(temperature,mass_n) # Electrons
    v_p = thermal_velocity(temperature, mass_p) # Holes

    # Trap time constants
    tau_p0 = 1/sigma_p*v_p*N_T
    tau_n0 = 1/sigma_n*v_n*N_T


if __name__ == "__main__":

    dataset = generate_dataset
    print("Stored dataset")
    data = pd.read.csv("datastet.csv")
    print(data.head)

    # # Creates or overwrites a new CSV file
    # with open('dataset.csv', 'w') as file:
    #     # Create a CSV writer object
    #     csv_writer = csv.writer(file)

    #     # Writing a single row
    #     csv_writer.writerow(["Name", "Age", "City"])

    #     # Read CSV data into a DataFrame
    #     data = pd.read_csv('example.csv')
    #     print(data)