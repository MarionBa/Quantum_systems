import numpy as np
import matplotlib.pyplot as plt
from ryd_numerov.rydberg import RydbergState

# Insert parameters of experiment in funciton, note that Omega_c/p are complex numbers in the tehoretical model
# however, upon solving, we end up with the square of this complex number aka square of absolute value

def Transmission(Delta_c,Delta_p,Omega_c,Omega_p,Gamma_e,Gamma_r,alpha, gamma):
    Im_rho_eg = []
    for elt in Delta_c:
        # The coupling field detuning
        Delta_c_idx = elt * (2 * np.pi)
        delta = -Delta_s + Delta_p + Delta_c_idx  # Two-photons detuning
        # Matrix to be solved, Eqs. 3 in steady state
        Matrix = [[1, 1, 1, 0, 0, 0, 0, 0, 0],
                   [0, -1j*Omega_c/2, 1j*Omega_c/2, 0, 0, 0, 1j*Omega_p/2, -((Gamma_e+Gamma_r)/2-1j*Delta_c_idx), 0],
                   [0, 0, -Gamma_r, 0, 0, 0, 0, 1j*Omega_c/2, -1j*Omega_c/2],
                   [0, 0, 0, 1j*Omega_c/2, 0, -(gamma+Gamma_r/2+1j*delta), 0, 0, -Omega_p/2],
                   [1j*Omega_p/2, -1j*Omega_p/2, 0, -(Gamma_e/2+1j*Delta_p), 0, 1j*Omega_c/2, 0, 0, 0],
                   [0, Gamma_e, 0, 1j*Omega_p/2, -1j*Omega_p/2, 0, 0, 0, 0],
                   [0, 1j*Omega_c/2, -1j*Omega_c/2, 0, 0, -1j*Omega_p/2, 0, 0, -((Gamma_e+Gamma_r)/2+1j*Delta_c_idx)],
                   [0, 0, -Gamma_r, 0, 0, 0, 0, -1j*Omega_c/2, 1j*Omega_c/2],
                   [-1j*Omega_p/2, 1j*Omega_p/2, 0, 0, -(Gamma_e/2-1j*Delta_p), 0, -1j*Omega_c/2, 0, 0]]
        Vector = [1, 0, 0, 0, 0, 0, 0, 0, 0]
        Solution = np.linalg.solve(Matrix, Vector)
        print(Solution)
        Im_rho_eg.append(np.imag((Solution[3]-Solution[4])/2))
    return np.exp(-alpha*Gamma_e*np.array(Im_rho_eg)/Omega_p)

def SigmaPerturbative(Delta_c,Delta_p,Omega_c,Gamma_e,alpha, gamma):
    Im_rho_eg = []
    for elt in Delta_c:
        Delta_c_idx = elt * (2 * np.pi)
        delta = Delta_s + Delta_p + Delta_c_idx  # Two-photons detuning
        # Matrix to be solved, Eqs. 3 in steady state
        Matrix = [[1j*Omega_c/2, -(Gamma_e/2+1j*Delta_p)],
                  [-(gamma+1j*delta), 1j*Omega_c/2]]
        Vector = [-1j*Omega_p/2, 0]
        Solution = np.linalg.solve(Matrix, Vector)
        Im_rho_eg.append(np.imag((Solution[1]-np.conjugate(Solution[1]))/2))
    return Im_rho_eg #np.exp((-alpha*Gamma_e*np.array(Im_rho_eg)/Omega_p))




""" First case: high intensity T=51degree """
Delta_c = np.linspace(-100, 500, 1000) # Coupling laser \Delta_c/2\pi [MHz]
alpha = 1.65 # Optical density [1]
Gamma_e = 2 * np.pi * 6 # Spontaneous emission excited state [MHz]
Gamma_r = 2 * np.pi * 5 # Spontaneous emission Rydberg state [MHz]
Omega_p = 2 * np.pi * 30 # The Rabi frequency of the probe laser [MHz]
print(Omega_p)
gamma = 2 * np.pi * 5.4 # Coherence dephasing rate between |g> and |r>
Delta_p = 0 # Detuning probe --> we take resonant case


# 5/2 state - |3, 8>
CG_3 = 1 * 1 * 1/3 # Hyper fine, fine and WE coef.
Omega_c = CG_3 * 2 * np.pi * 7.8 # The Rabi frequency of the coupling laser [MHz]
Delta_s = 2 * np.pi * 336.4 # EIT transition |33D_5/2> --> needs minus sign otherwise symetric at -336.4 MHz

# Transmission Rydberg #TransmissionPerturbative(Delta_c, Delta_p, Omega_c, Gamma_e, alpha, gamma)#
sigma_23 = SigmaPerturbative(Delta_c, Delta_p, Omega_c, Gamma_e, alpha, gamma)
sigma_78 = sigma_23

#######################################################################

# 3/2 state - |5, 10>
# Transmission Rydberg
CG_5 = 0.78/3#0.5 * 0.632 * 1/3 # Hyper fine, fine CGC and Wigner Eckart element respectively
Omega_c = CG_5 * 2 * np.pi * 7.8 # The Rabi frequency of the coupling laser [MHz]
Delta_s = 0 # EIT transition |33D_3/2>
sigma_25 = SigmaPerturbative(Delta_c, Delta_p, Omega_c, Gamma_e, alpha, gamma)
sigma_710 = sigma_25

#######################################################################

# 5/2 state - |4>
# Transmission Rydberg
CG_4 = 0.32/3#0.189 * 0.316 * 1/3
Omega_c = CG_4 * 2 * np.pi * 7.8 # The Rabi frequency of the coupling laser [MHz]
Delta_s = -2 * np.pi * 336.4 # EIT transition |33D_5/2> --> needs minus sign otherwise symetric at -336.4 MHz

sigma_24 = SigmaPerturbative(Delta_c, Delta_p, Omega_c, Gamma_e, alpha, gamma)
sigma_79 = sigma_24

# Total transmission
#T = (T_5 + T_4 + T_3)/2 + (T_8 + T_9 + T_10)/2
T = np.exp((-alpha*Gamma_e*np.array(sigma_23)*np.array(sigma_24)*np.array(sigma_25)/Omega_p))

#*np.array(sigma_24)*np.array(sigma_25)*np.array(sigma_78)*np.array(sigma_79)*np.array(sigma_710)

# Plot
plt.figure()
plt.plot(Delta_c, T*100)
plt.text(100, 27, r'$|33D_{5/2},mj=\pm5/2, mf=\pm4>$', fontsize=8)
plt.xlabel(r'$\Delta_c$/2$\pi$ [MHz]')
plt.ylabel(r'Transmission [%]')

plt.show()



