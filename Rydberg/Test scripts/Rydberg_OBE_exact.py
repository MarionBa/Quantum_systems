import numpy as np
import matplotlib.pyplot as plt

# Insert parameters of experiment in funciton, note that Omega_c/p are complex numbers in the tehoretical model
# however, upon solving, we end up with the square of this complex number aka square of absolute value

def Sigma(Delta_c,Delta_p, Delta_s,Delta_g,Delta_e,Delta_r,Omega_c,Omega_p,Gamma_e,Gamma_r,alpha, gamma, phi, B):
    Im_rho_eg = []
    for elt in Delta_c:
        Delta_c_idx = elt * (2 * np.pi)

        # Doppler parameter
        lambda_p = 780 * 10 ** -9 # [nm] unit does not matter as we take the ratio
        lambda_c = 480 * 10 ** -9 # [nm]
        Delta_D = Delta_c_idx #Delta_p+Delta_c_idx-(Delta_e-Delta_g) # 1 photon (Delta_e-Delta_g)*(lambda_p-lambda_c)/lambda_c
        delta = B*(Delta_g+Delta_e+Delta_r) +Delta_D - Delta_s + Delta_c_idx + Delta_p # Two-photons detuning
        # Matrix to be solved, Eqs. 3 in steady state #
        Matrix = [[1, 1, 1, 0, 0, 0, 0, 0, 0],
                   [0, Gamma_e, 0, 1j*Omega_p*np.exp(-1j*phi)/2, -1j*Omega_p*np.exp(1j*phi)/2, 0, 0, 0, 0],
                   [0, -Gamma_e, 0, -1j*Omega_p*np.exp(-1j*phi)/2, 1j*Omega_p*np.exp(1j*phi)/2, 0, 0, 1j*Omega_c*np.exp(-1j*phi)/2, -1j*Omega_c*np.exp(1j*phi)/2],
                   [0, 0, (1j*delta-Gamma_r), 0, 0, 0, 0, -1j*Omega_c*np.exp(-1j*phi)/2, 1j*Omega_c*np.exp(1j*phi)/2],
                   [1j*Omega_p*np.exp(1j*phi)/2, -1j*Omega_p*np.exp(1j*phi)/2, 0, (-1j*Delta_p-Gamma_e/2), 0, 1j*Omega_c*np.exp(-1j*phi)/2, 0, 0, 0],
                   [-1j*Omega_p*np.exp(-1j*phi)/2, 1j*Omega_p*np.exp(-1j*phi)/2, 0, 0, (1j*Delta_p-Gamma_e/2), 0, -1j*Omega_c*np.exp(1j*phi)/2, 0, 0],
                   [0, 0, 0, 1j*Omega_c*np.exp(1j*phi)/2, 0, -(gamma+Gamma_r/2), 0, -1j*Omega_p*np.exp(1j*phi)/2, 0],
                   [0, 0, 0, 0, -1j*Omega_c*np.exp(-1j*phi)/2, 0, -(gamma+Gamma_r/2), 0, 1j*Omega_p*np.exp(-1j*phi)/2],
                   [0, 1j*Omega_c*np.exp(1j*phi)/2, -1j*Omega_c*np.exp(1j*phi)/2, 0, 0, -1j*Omega_p*np.exp(-1j*phi)/2, 0, (1j*Delta_p-(Gamma_e+Gamma_r)/2), 0],
                   [0, -1j*Omega_c*np.exp(-1j*phi)/2, 1j*Omega_c*np.exp(-1j*phi)/2, 0, 0, 0, 1j*Omega_p*np.exp(1j*phi)/2, 0, (-1j*Delta_p-(Gamma_e+Gamma_r)/2)]]
        # Determinent calculation
        Vector = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Solution = np.linalg.lstsq(Matrix, Vector) #np.linalg.solve(Matrix, Vector)
        Sol = Solution[0]
        Im_rho_eg.append(np.imag((Sol[3]-Sol[4])/2))
    return Im_rho_eg # np.exp(-alpha*Gamma_e*np.array(Im_rho_eg)/Omega_p)


""" Test case """
Delta_c = np.linspace(-100, 500, 1000) # Coupling laser \Delta_c/2\pi [MHz]
alpha = 1.65 # Optical density [1]
Gamma_e = 2 * np.pi * 60 # Spontaneous emission excited state [MHz]
Gamma_r = 2 * np.pi * 5 # Spontaneous emission Rydberg state [MHz]
Omega_p = 2 * np.pi * 30# The Rabi frequency of the probe laser [MHz]
phi = 0 # phase of the electromagetic field with respect to the system
gamma = 2 * np.pi * 5.4 # Coherence dephasing rate between |g> and |r>
Delta_p = 0 # Detuning probe --> we take resonant case
# Magnetic parameters
gF_g = 1/2 # lande factor ground state
gF_e = 2/2 # lande factor excited state
mu_B_over_hbar = 1 # [MHz/G]
B = 10 # [Gauss] external magnetic field

################################################################

#### Our 5/2 states ####
# |3, 8>
CG_3 = 1 * 1 * 1/3 # Hyper fine, fine and WE coef.
Omega_c = CG_3 * 2 * np.pi * 7.8 # The Rabi frequency of the coupling laser [MHz]
Delta_s = 2 * np.pi * 336.4 # EIT transition |33D_5/2> --> needs minus sign otherwise symetric at -336.4 MHz
gF_r = 1.2

# Magnetic components |1> --> |2> --> |3> transition
mF_g = 2
Delta_g = mu_B_over_hbar * gF_g * mF_g
mF_e = 3
Delta_e = mu_B_over_hbar * gF_e * mF_e
mF_r = 4
Delta_r = mu_B_over_hbar * gF_r * mF_r

# Transmission Rydberg #TransmissionPerturbative(Delta_c, Delta_p, Omega_c, Gamma_e, alpha, gamma)#
sigma_23 = Sigma(Delta_c,Delta_p, Delta_s,Delta_g,Delta_e,Delta_r,Omega_c,Omega_p,Gamma_e,Gamma_r,alpha, gamma, phi, B)

# Magnetic components |6> --> |7> --> |8> transition
mF_g = -2
Delta_g = mu_B_over_hbar * gF_g * mF_g
mF_e = -3
Delta_e = mu_B_over_hbar * gF_e * mF_e
mF_r = -4
Delta_r = mu_B_over_hbar * gF_r * mF_r

# Transmission Rydberg
sigma_78 = Sigma(Delta_c,Delta_p, Delta_s,Delta_g,Delta_e,Delta_r,Omega_c,Omega_p,Gamma_e,Gamma_r,alpha, gamma, phi, B)

#######################################################################

#### Our 3/2 states ####
# |5, 10>
CG_5 = 0.78/3#0.5 * 0.632 * 1/3 # Hyper fine, fine CGC and Wigner Eckart element respectively
Omega_c = CG_5 * 2 * np.pi * 7.8 # The Rabi frequency of the coupling laser [MHz]
Delta_s = 0 # EIT transition |33D_3/2>
gF_r = 0.8

# Magnetic components |1> --> |2> --> |5> transition
mF_g = 2
Delta_g = mu_B_over_hbar * gF_g * mF_g
mF_e = 3
Delta_e = mu_B_over_hbar * gF_e * mF_e
mF_r = 2
Delta_r = mu_B_over_hbar * gF_r * mF_r

# Transmission Rydberg #TransmissionPerturbative(Delta_c, Delta_p, Omega_c, Gamma_e, alpha, gamma)#
sigma_25 = Sigma(Delta_c,Delta_p, Delta_s,Delta_g,Delta_e,Delta_r,Omega_c,Omega_p,Gamma_e,Gamma_r,alpha, gamma, phi, B)

# Magnetic components |6> --> |7> --> |10> transition
mF_g = -2
Delta_g = mu_B_over_hbar * gF_g * mF_g
mF_e = -3
Delta_e = mu_B_over_hbar * gF_e * mF_e
mF_r = -2
Delta_r = mu_B_over_hbar * gF_r * mF_r

# Transmission Rydberg #TransmissionPerturbative(Delta_c, Delta_p, Omega_c, Gamma_e, alpha, gamma)#
sigma_710 = Sigma(Delta_c,Delta_p, Delta_s,Delta_g,Delta_e,Delta_r,Omega_c,Omega_p,Gamma_e,Gamma_r,alpha, gamma, phi, B)


#######################################################################

#### Our 5/2 states ####
# |4>, |9>
CG_4 = 0.32/3#0.189 * 0.316 * 1/3
Omega_c = CG_4 * 2 * np.pi * 7.8 # The Rabi frequency of the coupling laser [MHz]
Delta_s = 2 * np.pi * 336.4 # EIT transition |33D_5/2> --> needs minus sign otherwise symetric at -336.4 MHz
gF_r = 1.2

# Magnetic components |1> --> |2> --> |4> transition
mF_g = 2
Delta_g = mu_B_over_hbar * gF_g * mF_g
mF_e = 3
Delta_e = mu_B_over_hbar * gF_e * mF_e
mF_r = 2
Delta_r = mu_B_over_hbar * gF_r * mF_r

# Transmission Rydberg #TransmissionPerturbative(Delta_c, Delta_p, Omega_c, Gamma_e, alpha, gamma)#
sigma_24 = Sigma(Delta_c,Delta_p, Delta_s,Delta_g,Delta_e,Delta_r,Omega_c,Omega_p,Gamma_e,Gamma_r,alpha, gamma, phi, B)

# Magnetic components |1> --> |2> --> |4> transition
mF_g = -2
Delta_g = mu_B_over_hbar * gF_g * mF_g
mF_e = -3
Delta_e = mu_B_over_hbar * gF_e * mF_e
mF_r = -2
Delta_r = mu_B_over_hbar * gF_r * mF_r

# Transmission Rydberg #TransmissionPerturbative(Delta_c, Delta_p, Omega_c, Gamma_e, alpha, gamma)#
sigma_79 = Sigma(Delta_c,Delta_p, Delta_s,Delta_g,Delta_e,Delta_r,Omega_c,Omega_p,Gamma_e,Gamma_r,alpha, gamma, phi, B)


# Total transmission
T = np.exp(-alpha*Gamma_e*np.array(sigma_23)*np.array(sigma_24)*np.array(sigma_25)/Omega_p)


# Plot
plt.figure()
plt.plot(Delta_c, T*100)
plt.text(100, 27, r'$|33D_{5/2},mj=\pm5/2, mf=\pm4>$', fontsize=8)
plt.xlabel(r'$\Delta_c$/2$\pi$ [MHz]')
plt.ylabel(r'Transmission [%]')

plt.show()






