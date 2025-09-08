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


M_E  = 9.10938356e-31 # Electron effective mass
# Si
MASS_N = 1.08* M_E # Effective mass of electrons
MASS_P = 0.81 * M_E # Effective mass of holes


def thermal_velocity(T,eff_m):
    "Computes thermal velocity according to Green's model"
    return np.sqrt((3* K_B_J*T)/eff_m)

def Nc(T):
    """Effective density of states in the conduction band
    T: temperature (kelvin)
    Divided by 10^6 as we want cm^-3"""
    return (2*((3*np.pi*MASS_N*K_B_J*T)/h**2)**(3/2))/1e6

def Nv(T):
    """Effective density of states in the valence band
    T: temperature (kelvin)
    Divided by 10^6 as we want cm^-3"""
    return (2*((3*np.pi*MASS_P*K_B_J*T)/h**2)**(3/2))/1e6


def compute_srh(temperature, delta_n, e_t, sigma_n, sigma_p, N_T=1e12, N_DOP = 5.1e15):
    """Computes SRH formula
    delta_n <- Excess carrier transformation
    e_t <- Defect energy level
    sigma_n, sigma_p <- Capture cross sections
    N_T <- Defect Density
    N_DOP <- Doping density"""

    # Thermal velocities, in cm/s
    v_n = thermal_velocity(temperature,MASS_N) * 100 # Electrons
    v_p = thermal_velocity(temperature, MASS_P) * 100 # Holes

    # Intrinsic carrier concentration
    nc = Nc(temperature)
    nv = Nv(temperature)
    n_i = np.sqrt(nc*nv)*np.exp(-E_G_SI/(2*K_B_EV*temperature))
    
    # n_0 and p_0
    n_0 = n_i**2/N_DOP # Electron carrier concentration at thermal equilibrium
    p_0 = N_DOP # Hole "" "" "" "" ""
    
    # n1 and p1
    n_1 = n_i * np.exp(e_t/(K_B_EV*temperature))
    p_1 = n_i * np.exp(-e_t/(K_B_EV*temperature))

    # Trap time constants
    tau_p0 = 1/(sigma_p*v_p*N_T)
    tau_n0 = 1/(sigma_n*v_n*N_T)

    # SRH formula
    numerator = tau_p0*(n_0 + n_1 + delta_n) + tau_n0*(p_0 + p_1+ delta_n)
    denominator = n_0 + p_0 + delta_n
    srh = numerator/denominator

    # Adding optional Gaussian noise
    # if NOISE > 0:
    #       srh *= 1 + np.random.normal(0,NOISE)

    # Returning all values as we wish to store them for future use
    #return srh, tau_n0, tau_p0, n_1, p_1,n_0,p_0,n_i,nc,nv, v_n, v_p
    return srh

def compute_srh_two_defects(temperature, delta_n, e_t, sigma_n, sigma_p, e_t_2,sigma_n_2,sigma_p_2):
    """Computes the Tau SRH of a curve with two defects. Considering the formula is
    1/tau_total = 1/first_tau + 1/second_tau
    temperature
    delta_n <- Excess carrier concentration
    e_t, sigma_n, sigma_p <- First defect combination parameters
    e_t_2, sigma_n_2, sigma_p_2 <- Second defect combination parameters
    N_T"""

    # First component
    first_tau = compute_srh(temperature, delta_n, e_t, sigma_n, sigma_p, N_T,N_DOP)
    
    # Second component
    second_tau = compute_srh(temperature, delta_n, e_t_2, sigma_n_2, sigma_p_2,N_T,N_DOP)

    # Operation
    computation = 1/first_tau + 1/second_tau

    # Resulting SRH
    return 1/computation

def compute_srh_multilevel(temperature, delta_n, e_t, sigma_n, sigma_p, e_t_2,sigma_n_2,sigma_p_2):
    """Function which uses generalized Shockley-Red-Hall SRH lifetime formula
    temperature
    delta_n <- Excess carrier concentration
    e_t, sigma_n, sigma_p <- First defect parameter combinations
    e_t_2, sigma_n_2, sigma_p_2 <- Second defect parameter combination"""

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

    # n_0 and p_0
    n_0 = n_i**2/N_DOP
    p_0 = N_DOP
    
    # n and p
    n = n_0 + delta_n
    p = p_0 + delta_n



    # n1 and p1
    n_1 = n_i * np.exp(e_t/(K_B_EV*temperature))
    p_1 = n_i * np.exp(-e_t/(K_B_EV*temperature))

    # n2 and p2
    n_2 = n_i * np.exp((e_t_2)/(K_B_EV*temperature))
    p_2 = n_i * np.exp((-e_t_2)/(K_B_EV*temperature))

    
    # Applying the formula
    numerator_part1 = 1 +(sigma_n*v_n*n_1 + sigma_p*v_p*p)/(sigma_p*v_p*p_1 + sigma_n*v_n*n)
    numerator_part2 = (sigma_p_2*v_p*p_2 + sigma_n_2*v_n*n)/(sigma_n_2*v_n*n_2 + sigma_p_2*v_p*p)
    numerator = numerator_part1 + numerator_part2

    denominator_part1 = N_T*(n_0 + p_0 + delta_n)
    denominator_part2 = (sigma_n*sigma_p*v_p*v_n)/(sigma_p*v_p*p_1 + sigma_n*v_n*n)
    denominator_part3 = (sigma_n_2*sigma_p_2*v_p*v_n)/(sigma_n_2*v_n*n_2 + sigma_p_2*v_p*p)

    # Final computation
    denominator = denominator_part1*(denominator_part2+denominator_part3)

    return numerator/denominator


def store_curves(temperatures, n_points, defect_type,start_curve_id:int):
    """Functions which creates curves
    temperatures <- list of temperatures
    n_points <- Points to represent each curve
    defect_type <- Defects or labels {one,two,two_levels} It can be one element, two elements, or an element with two defects
    start_curve_id <- Identifier for each curve"""

    # Initializing excess carrier concentration
    delta_ns = np.logspace(13,17,n_points)
    
    # Defect level combinations 
    e_t = np.random.uniform(-0.55,0.55)
    sigma_n = 10**np.random.uniform(-17,-13)
    sigma_p = 10**np.random.uniform(-17,-13)

    # Second defect level combinations (For multi defect components)
    e_t_2 = np.random.uniform(-0.55,0.55)
    sigma_n_2 = 10**np.random.uniform(-17,-13)
    sigma_p_2 = 10**np.random.uniform(-17,-13)

    # Initializing empty dataset
    rows = []
    curve_id = start_curve_id # Updating curve id


    # Combination for all temperatures
    for t in temperatures:         
            # Computing SRH and storing the result
            for delta in delta_ns:
                """srh_sum,tau_n0_sum, tau_p0_sum, n_1_sum, p_1_sum,n_0_sum,p_0_sum,n_i_sum,nc_sum,nv_sum, v_n_sum, v_p_sum = 0,0,0,0,0,0,0,0,0,0,0,0"""
                
                if defect_type == "one":
                    srh = compute_srh(t, delta, e_t, sigma_n, sigma_p)
                    """#(srh_i, tau_n0_i, tau_p0_i, n_1_i, p_1_i, n_0_i,p_0_i ,n_i_i,nc_i,nv_i, v_n_i, v_p_i) = result"""
                    label = 0

                elif defect_type == "two":
                    srh = compute_srh_two_defects(t, delta, e_t, sigma_n, sigma_p,e_t_2,sigma_n_2,sigma_p_2)
                    label = 1

                elif defect_type == "two_levels":
                    srh = compute_srh_multilevel(t, delta, e_t, sigma_n, sigma_p, e_t_2,sigma_n_2,sigma_p_2)
                    label = 2 


                rows.append({"Δn": delta, "T": t,"SRH":srh,  "curve_id": curve_id,"Label": label})
                

                """rows.append({"Δn": delta, "T": t,"SRH": srh_sum, "𝜏_n0":tau_n0_sum, "𝜏_p0":tau_p0_sum, "n_1":n_1_sum,
                             "p_1": p_1_sum, "n_0":n_0_sum, "p_0": p_0_sum, "n_i": n_i_sum, "Nc": nc_sum, "Nv": nv_sum, "V_n":v_n_sum,
                             "V_p":v_p_sum, "curve_id": curve_id,
                             "Label": label})"""

            # Updating curve id
            curve_id += 1
            

    # Storing the curve
    df = pd.DataFrame(rows)
    

    return df


def save_curve_image(df, curve_id, label, out_dir = "curve_images"):
    """Function to obtain images from the generated curves
    df <- Dataframe containing the points used for generating the curves
    curve_id <- Curve identifier
    label <- Label
    out_dir <- folder for storing images"""

    os.makedirs(out_dir, exist_ok= True)

    fig, ax = plt.subplots( figsize = (6,4))
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(df["Δn"], df["SRH"])
        
    ax.set_xlabel('Excess carrier concentration')
    ax.set_ylabel('Lifetime (SRH)')
    ax.set_title('SRH Lifetime Curve', size = 14)
    
    filename =  os.path.join(out_dir,f"image_{curve_id}Label_{label}.png")
    plt.savefig(filename, dpi = 150, bbox_inches = "tight")
    plt.close(fig)

    


if __name__ == "__main__":
    
    # Initializing temperatures
    #temperatures = [200,225,250,275,300,325,350,375,400]
    temperatures = [300] # Just to start

    # Verifying the desired path exists
    path = "./metadata.csv"
    exists = os.path.isfile(path)

    # If it exists, verify the amount of rows to update curve_id
    if exists:
        data = pd.read_csv("metadata.csv", header = 0)
        # No data
        if data.empty:
            curve_id = 0
        
        # There is data, update curve_id
        else: 
            curve_id = data["curve_id"].max() + 1

    else:
        curve_id = 0


    samples = 10000 # Curves per group 
    dataset = []
    defect_type = ["one","two", "two_levels"]

    # Computing the curves
    for defect_types in defect_type:
        for i in range(samples):
            df = store_curves(temperatures, n_points = 100, defect_type=defect_types, start_curve_id=curve_id)
            dataset.append(df)
            print("Row:",i)
            curve_id = df["curve_id"].max() + 1

    # Saving the data
    curves = pd.concat(dataset,ignore_index=True) # Ignore index avoids potential duplicate indices

    # If dataset exists
    if os.path.exists("metadata.csv"):
        curves.to_csv("metadata.csv", index = False, mode = "a", header = False) # Append without the header
    else:
        curves.to_csv("metadata.csv", index = False) # Add with the header


    # Storing data as images
    # for ids in curves["curve_id"].unique():
    #             subset = curves[curves["curve_id"]==ids]
    #             label = subset["Label"].iloc[0]
    #             save_curve_image(subset,ids,label, out_dir="curve_images")   

    

    