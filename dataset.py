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
    T: temperature (kelvin)
    Divided by 10^6 as we want cm^-3"""
    return (2*((3*np.pi*mass_n*K_B_J*T)/h**2)**(3/2))/1e6


def Nv(T):
    """Effective density of states in the valence band
    T: temperature (kelvin)
    Divided by 10^6 as we want cm^-3"""
    return (2*((3*np.pi*mass_p*K_B_J*T)/h**2)**(3/2))/1e6


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

    # Thermal velocities, in cm/s
    v_n = thermal_velocity(temperature,mass_n) * 100 # Electrons
    v_p = thermal_velocity(temperature, mass_p) * 100 # Holes

    # Intrinsic carrier concentration
    nc = Nc(temperature)
    nv = Nv(temperature)
    n_i = np.sqrt(nc*nv)*np.exp(-E_G_SI/(2*K_B_EV*temperature))
    
    n_0 = n_i**2/N_DOP
    p_0 = N_DOP
    
    # n1 and p1
    n_1 = n_i * np.exp(e_t/(K_B_EV*temperature))
    p_1 = n_i * np.exp(-e_t/(K_B_EV*temperature))

    # Trap time constants
    tau_p0 = 1/(sigma_p*v_p*N_T)
    tau_n0 = 1/(sigma_n*v_n*N_T)

    # SRH
    numerator = tau_p0*(n_0 + n_1 + delta_n) + tau_n0*(p_0 + p_1+ delta_n)
    denominator = n_0 + p_0 + delta_n
    srh = numerator/denominator

    return srh

def store_curves(temperatures, n_points, defect_type):
    """Functions which creates curves
    temperatures <- list of temperatures
    n_points <- Points to represent each curve
    defect_type <- Defects {one,two,two_levels}"""

    # Initializing excess carrier concentration
    delta_ns = np.logspace(13,17,n_points)

    # Assigning defects depending on the defect type
    if defect_type == "one":
        n_defects = 1

    elif defect_type == "two":
        n_defects = 2

    elif defect_type == "two_levels":
        n_defects = 2 # One element two levels

    else:
        raise ValueError("Invalid type")
    
    
    # Defect level combinations 
    e_t = np.random.uniform(-0.55,0.55,n_defects)
    sigma_n = 10**np.random.uniform(-17,-13)
    sigma_p = 10**np.random.uniform(-17,-13)

    # Initializing empty dataset
    rows = []

    # Computing parameters
    for t in temperatures:         
            # Computing srh and storing the result
            for delta in delta_ns:
                srh_sum = 0

                for i in range(n_defects):
                    srh_sum += compute_srh(t, delta, e_t[i], sigma_n[i], sigma_p[], N_T=1e12, N_DOP = 5.1e15)
                
                # Labelling our data
                if defect_type == "one":
                    label = 0
                
                elif defect_type == "two":
                    label = 1

                elif defect_type == "two_levels":
                    label = 2


                rows.append({"Δn": delta, "T": t,"SRH": srh_sum, "Label": label})
    
    df = pd.DataFrame(rows)

    return df

def show_curve():
    
    df = pd.read_csv("srh_curves.csv")

    fig, ax = plt.subplots( figsize = (9,6))
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Obtaining the set of different temperatures
    temperatures = sorted(df["T"].unique())

    # Plotting the curve
    for T in temperatures:
        subset = df[df["T"] == T]

        ax.plot(subset["Δn"], subset["SRH"], label=f'T={T}K')
    
    
    
    ax.set_xlabel('Excess carrier concentration')
    ax.set_ylabel('Lifetime (SRH)')
    ax.set_title('SRH Lifetime Curve', size = 14)
    plt.show()


if __name__ == "__main__":
    
    # Initializing temperatures
    temperatures = [200,225,250,275,300,325,350,375,400]

    # Each defect combination is simulated with 9 different temperatures2
    curves = []

    # Samples per class
    samples = 1
    for _ in range(samples):
        curves.append(store_curves(temperatures,n_points=100,defect_type="one"))
        curves.append(store_curves(temperatures,n_points=100,defect_type="two"))
        curves.append(store_curves(temperatures,n_points=100,defect_type="two_levels"))

    df = pd.concat(curves, ignore_index= False)
    df.to_csv("srh_curves.csv", index=False)
    # df.to_csv("srh_curves.csv", index=False, mode = "a")

    # Plotting the curve
    #show_curve()

    # cuando veo una curva quiero saber si una impureza con dos niveles o dos impurezas <- supervised
    # si es concava una impureza, convexa unknown <- clustering
    

# Preguntas

# En la grafica, eso es la misma curva o no?

# Supervisado o no supervisado