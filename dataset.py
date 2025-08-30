import os
import csv
import pandas as pd
import numpy as np
import random 
import matplotlib.pyplot as plt

# Constants

K_B_J = 1.380649e-23 # Boltzman Constant J/K
K_B_EV = 8.61733e-5 # Boltzman Constant EV
h = 6.626e-34 # Planck constant
N_T = 1e12 # Defect density
N_DOP = 5.1e15 # Doping density = p_0
DOPING_TYPE = "p"
NOISE = 0.01
E_G_SI = 1.1 # Bandgap energy silicon

# Masses
M_E  = 9.10938356e-31 # Electron effective mass

# For Si
mass_n = 1.08* M_E # Effective mass of electrons
mass_p = 0.81 * M_E # Effective mass of holes


def thermal_velocity(T,eff_m):
    "Computes thermal velocity according to Green's model"
    return np.sqrt((3* K_B_J*T)/eff_m)


# Obtaining different elements and formulas to clarify code
def Nc(T):
    """Effective density of states in the conduction band
    T: temperature (kelvin)"""
    return 2*((3*np.pi*mass_n*K_B_J*T)/h**2)**(3/2)


def Nv(T):
    """Effective density of states in the valence band
    T: temperature (kelvin)"""
    return 2*((3*np.pi*mass_p*K_B_J*T)/h**2)**(3/2)


# Generating dataset
# def generate_dataset(N):
#     """Generates a dataset of N rows"""
#     e_t = np.random.uniform(-0.55,0.55, N) # Defect Energy Level
#     sigma_p = 10**np.random.uniform(-17,-13,N) # Capture cross sections p
#     sigma_n = 10**np.random.uniform(-17,-13,N) # Capture cross section n
#     k = sigma_n/sigma_p # Capture cross section ratio
#     delta_n = 10**np.random.uniform(1e13, 1e17, N) # Excess carrier concentration
#     temperature = np.random.uniform(200,400, N)
#     alpha = np.random.uniform(1.1,4, N) # Power law coefficient
#     e_v = np.random.uniform(0.01,0.25) # Activation Energy
#     p_0 = N_DOP

#     # Thermal velocities
#     v_n = thermal_velocity(temperature,mass_n) # Electrons
#     v_p = thermal_velocity(temperature, mass_p) # Holes

#     # Intrinsic carrier concentration
#     nc = Nc(temperature)
#     nv = Nv(temperature)
#     n_i = np.sqrt(nc*nv)*np.exp(-E_G_SI/(2*K_B_J*temperature))

#     n_0 = n_i**2/N_DOP
#     p_0 = N_DOP

#     # n1 and p1
#     n_1 = n_i * np.exp(e_t/(K_B_EV*temperature))
#     p_1 = n_i * np.exp(-e_t/(K_B_EV*temperature))

#     # Trap time constants
#     tau_p0 = 1/sigma_p*v_p*N_T
#     tau_n0 = 1/sigma_n*v_n*N_T

#     # SRH
#     numerator = tau_p0*(n_0 + n_1 + delta_n) + tau_n0*(p_0 + p_1+ delta_n)
#     denominator = n_0 + p_0 + delta_n
#     srh = numerator/denominator

#     # Generating the dataset
#     df = pd.DataFrame({"E_t": e_t, "CS_n": sigma_n, "CS_p": sigma_p, "CS_ratio": k, "Δn": delta_n, "T": temperature,"α": alpha, "E_v": e_v, "SRH": srh })

#     return df
 

# def show_curves():
#     """Function to represent the curves from the dataset"""
#     df = generate_dataset(20)

#     df.plot(x="Δn", y = "SRH", logy=True)

#     plt.show()

def compute_srh(temperature, delta_n, e_t, sigma_n, sigma_p, N_T=1e12, N_DOP = 5.1e15):
    """Computes SRH formula
    delta_n <- Excess carrier transformation
    E_t <- Defect energy level
    sigma_n, sigma_p <- Capture cross sections
    N_T <- Defect Density
    N_DOP <- Doping density"""

    # Constants
    K_B_EV = 8.61733e-5 # Boltzman Constant EV
    h = 6.626e-34 # Planck constant
    E_G_SI = 1.1 # Bandgap energy silicon
    M_E  = 9.10938356e-31 # Electron effective mass

    # For Si
    mass_n = 1.08* M_E # Effective mass of electrons
    mass_p = 0.81 * M_E # Effective mass of holes

    # Thermal velocities
    v_n = thermal_velocity(temperature,mass_n) # Electrons
    v_p = thermal_velocity(temperature, mass_p) # Holes

    # Intrinsic carrier concentration
    nc = Nc(temperature)
    nv = Nv(temperature)
    n_i = np.sqrt(nc*nv)*np.exp(-E_G_SI/(2*K_B_J*temperature))

    n_0 = n_i**2/N_DOP
    p_0 = N_DOP
    
    # n1 and p1
    n_1 = n_i * np.exp(e_t/(K_B_EV*temperature))
    p_1 = n_i * np.exp(-e_t/(K_B_EV*temperature))

    # Trap time constants
    tau_p0 = 1/sigma_p*v_p*N_T
    tau_n0 = 1/sigma_n*v_n*N_T

    # SRH
    numerator = tau_p0*(n_0 + n_1 + delta_n) + tau_n0*(p_0 + p_1+ delta_n)
    denominator = n_0 + p_0 + delta_n
    srh = numerator/denominator

    return srh

def store_curves(temperature, n_points, curves):
    """Functions which creates curves
    n_points <- Points to represent each curve
    curves <- amount of curves to create"""

    # Initializing excess carrier concentration
    delta_ns = np.logspace(13,17,n_points)

    # Defect level
    e_t = []
    # sigmas
    sigma_p = []
    sigma_n = []

    for i in range(curves):
        e_t.append(np.random.uniform(-0.55,0.55))
        sigma_n.append(10**np.random.uniform(-17,-13))
        sigma_p.append(10**np.random.uniform(-17,-13))

    # Initializing empty dataset
    df = []

    # Computing parameters
    for i in range(curves):
        de = e_t[i]
        sigman = sigma_n[i]
        sigmap = sigma_p[i]

        for delta in delta_ns:
            srh = compute_srh(temperature, delta_ns, de, sigman, sigmap, N_T=1e12, N_DOP = 5.1e15)

            df.append({"E_t": e_t, "CS_n": sigma_n, "CS_p": sigma_p,"Δn": delta_ns, "T": temperature,"SRH": srh })
    
    return pd.DataFrame(df)


if __name__ == "__main__":

    df = store_curves(temperature=300, n_points=200, curves=2)

    df.to_csv("srh_curves.csv", index=False)
    # dataset = generate_dataset(20)
    #show_curves()
    # print("Stored dataset")
    # dataset.to_csv("dataset.csv", header = False, mode = "a", index = False)
    #dataset.to_csv("dataset.csv", index = False)
    # If we wish to append a new line, include variable mode
    