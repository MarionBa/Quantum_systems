import numpy as np
import matplotlib.pyplot as plt

# Insert parameters of experiment in funciton, note that Omega_c/p are complex numbers in the tehoretical model
# however, upon solving, we end up with the square of this complex number aka square of absolute value

def Sigma(Delta_c,Delta_p, Delta_s,Delta_g,Delta_e,Delta_r,Omega_c,Omega_p,Gamma_e,Gamma_r, gamma, phi, B):
    Im_rho_eg = []
    for elt in Delta_c:
        Delta_c_idx = elt * (2 * np.pi)

        # Doppler parameter
        lambda_p = 780 * 10 ** -9 # [nm] unit does not matter as we take the ratio
        lambda_c = 480 * 10 ** -9 # [nm]
        Delta_D = (Delta_e-Delta_g)*(lambda_p-lambda_c)/lambda_c
        delta = B*(Delta_g+Delta_e+Delta_r+Delta_D) + Delta_s + Delta_c_idx # Two-photons detuning
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
    return Gamma_e * np.array(Im_rho_eg) / Omega_p

def SigmaPerturbative(Delta_c,Delta_p, Delta_g, Delta_e, Delta_r, Delta_s,Omega_c, Omega_p,Gamma_e, gamma, B):
    Im_rho_eg = []
    for elt in Delta_c:
        Delta_c_idx = elt * (2 * np.pi)
        lambda_p = 780 * 10 ** -9  # [nm] unit does not matter as we take the ratio
        lambda_c = 480 * 10 ** -9  # [nm]
        Delta_D = (Delta_e - Delta_g) * (lambda_p - lambda_c) / lambda_c
        delta = B * (Delta_g + Delta_e + Delta_r + Delta_D) + Delta_s + Delta_c_idx  # Two-photons detuning
        #delta = -Delta_s + Delta_p + Delta_c_idx
        # Matrix to be solved, Eqs. 3 in steady state
        Matrix = [[1j*Omega_c/2, -(Gamma_e/2+1j*Delta_p)],
                  [-(gamma+1j*delta), 1j*Omega_c/2]]
        Vector = [-1j*Omega_p/2, 0]
        Solution = np.linalg.solve(Matrix, Vector)
        Im_rho_eg.append(np.imag((Solution[1]-np.conjugate(Solution[1]))/2))
    return Gamma_e * np.array(Im_rho_eg) / Omega_p


def SigmaPerturbativeAnalytic(Delta_c,Delta_p, Delta_g, Delta_e, Delta_r, Delta_s, Omega_c ,Gamma_e, gamma):
    sigma = []
    for elt in Delta_c:
        Delta_c_idx = elt * (2 * np.pi)
        lambda_p = 780 * 10 ** -9  # [nm] unit does not matter as we take the ratio
        lambda_c = 480 * 10 ** -9  # [nm]
        Delta_D = (Delta_e - Delta_g) * (lambda_p - lambda_c) / lambda_c
        delta = B * (Delta_g + Delta_e + Delta_r + Delta_D) + Delta_s + Delta_c_idx  # Two-photons detuning
        sigma.append((Gamma_e ** 2 * (4 * gamma ** 2 + 4 * delta ** 2) + 2 * gamma * Omega_c ** 2 * Gamma_e) / (
                (2 * gamma * Gamma_e + Omega_c ** 2 - 4 * Delta_p * delta) ** 2 + (
                    4 * Delta_p * gamma + 2 * Gamma_e * delta) ** 2))
    return np.array(sigma)

""" Magnetic field case """

Delta_c = np.linspace(-100, 500, 1000) # Coupling laser \Delta_c/2\pi [MHz]
Boost = 7.8
alpha = 0.36 # Optical density [1]
Gamma_e = 2 * np.pi * 60 # Spontaneous emission excited state [MHz]
Gamma_r = 2 * np.pi * 5 # Spontaneous emission Rydberg state [MHz]
Omega_p = 2 * np.pi * 30 # The Rabi frequency of the probe laser [MHz]
Omega_c_CGC1 = Boost * 2 * np.pi * 2.1 # The Rabi frequency of the coupling laser for CGC = 1 [MHz]
phi = 0 # phase of the electromagetic field with respect to the system
gamma = 2 * np.pi * 6 #5.4 # Coherence dephasing rate between |g> and |r>
Delta_p = 0 # Detuning probe --> we take resonant case

# Magnetic parameters
gF_g = 1/2 # lande factor ground state
gF_e = 2/3 # lande factor excited state
mu_B_over_hbar = 1 # [MHz/G]
B = 22 # [Gauss] external magnetic field

################################################################

#### Our 5/2 states ####
# |3, 8>
CG_3 = 1 * 2/3 # Hyper fine, fine and WE coef.
Emp_fact = 1.5 # Empirical factor for fitting experimental data
Omega_c = CG_3 * Emp_fact * Omega_c_CGC1 # The Rabi frequency of the coupling laser [MHz]
Delta_s = -2 * np.pi * 336.4 # EIT transition |33D_5/2> --> needs minus sign otherwise symetric at -336.4 MHz
gF_r = 3 * 1.2

# Magnetic components |1> --> |2> --> |3> transition
mF_g = 2
Delta_g = mu_B_over_hbar * gF_g * mF_g
mF_e = 3
Delta_e = mu_B_over_hbar * gF_e * mF_e
mF_r = 4
Delta_r = mu_B_over_hbar * gF_r * mF_r

# Transmission Rydberg ##
sigma_23 = Sigma(Delta_c,Delta_p,Delta_g,Delta_e,Delta_r, Delta_s,Omega_c,Omega_p,Gamma_e,Gamma_r, gamma, phi, B)
#sigma_23 = SigmaPerturbative(Delta_c, Delta_p, Delta_g, Delta_e, Delta_r, Delta_s, Omega_c,Omega_p, Gamma_e, gamma, B)
#sigma_23 = SigmaPerturbativeAnalytic(Delta_c, Delta_p, Delta_g, Delta_e, Delta_r, Delta_s, Omega_c, Omega_p, Gamma_e, gamma)


# Magnetic components |6> --> |7> --> |8> transition
mF_g = -2
Delta_g = mu_B_over_hbar * gF_g * mF_g
mF_e = -3
Delta_e = mu_B_over_hbar * gF_e * mF_e
mF_r = -4
Delta_r = mu_B_over_hbar * gF_r * mF_r

# Transmission Rydberg
sigma_78 = Sigma(Delta_c,Delta_p,Delta_g,Delta_e,Delta_r, Delta_s,Omega_c,Omega_p,Gamma_e,Gamma_r, gamma, phi, B)
#sigma_78 = SigmaPerturbative(Delta_c, Delta_p, Delta_g, Delta_e, Delta_r, Delta_s, Omega_c,Omega_p, Gamma_e, gamma, B)
#sigma_78 = SigmaPerturbativeAnalytic(Delta_c, Delta_p, Delta_g, Delta_e, Delta_r, Delta_s, Omega_c, Omega_p, Gamma_e, gamma)


#######################################################################

#### Our 3/2 states ####
# |5, 10>
CG_5 = 0.78 * 1/3 #0.5 * 0.632 * 1/3 # Hyper fine, fine CGC and Wigner Eckart element respectively
Omega_c = CG_5 * Omega_c_CGC1 # The Rabi frequency of the coupling laser [MHz]
Delta_s = 0 # EIT transition |33D_3/2>

# Ad Hoc parameters for no splitting of 3/2 peak as in ref. paper
gF_r = 0.8

# Magnetic components |1> --> |2> --> |5> transition
mF_g = 2
Delta_g = 0*mu_B_over_hbar * gF_g * mF_g
mF_e = 3
Delta_e = 0*mu_B_over_hbar * gF_e * mF_e
mF_r = 2
Delta_r = 0*mu_B_over_hbar * gF_r * mF_r

# Transmission Rydberg
sigma_25 = Sigma(Delta_c,Delta_p,Delta_g,Delta_e,Delta_r, Delta_s,Omega_c,Omega_p,Gamma_e,Gamma_r, gamma, phi, B)
#sigma_25 = SigmaPerturbative(Delta_c, Delta_p, Delta_g, Delta_e, Delta_r, Delta_s, Omega_c,Omega_p, Gamma_e, gamma, B)
#sigma_25 = SigmaPerturbativeAnalytic(Delta_c, Delta_p, Delta_g, Delta_e, Delta_r, Delta_s, Omega_c, Omega_p, Gamma_e, gamma)


# Magnetic components |6> --> |7> --> |10> transition
mF_g = -2
Delta_g = 0*mu_B_over_hbar * gF_g * mF_g
mF_e = -3
Delta_e = 0*mu_B_over_hbar * gF_e * mF_e
mF_r = -2
Delta_r = 0*mu_B_over_hbar * gF_r * mF_r

# Transmission Rydberg
sigma_710 = Sigma(Delta_c,Delta_p,Delta_g,Delta_e,Delta_r, Delta_s,Omega_c,Omega_p,Gamma_e,Gamma_r, gamma, phi, B)
#sigma_710 = SigmaPerturbative(Delta_c, Delta_p, Delta_g, Delta_e, Delta_r, Delta_s, Omega_c,Omega_p, Gamma_e, gamma, B)
#sigma_710 = SigmaPerturbativeAnalytic(Delta_c, Delta_p, Delta_g, Delta_e, Delta_r, Delta_s, Omega_c, Omega_p, Gamma_e, gamma)



#######################################################################

#### Our 5/2 states ####
# |4>, |9>
CG_4 = 0.32 * 2/3   #0.189 * 0.316 * 1/3
Emp_fact = 1.5 # Empirical factor for fitting experimental data
Omega_c = CG_4 * Emp_fact * Omega_c_CGC1 # The Rabi frequency of the coupling laser [MHz]
Delta_s = -2 * np.pi * 336.4 # EIT transition |33D_5/2> --> needs minus sign otherwise symetric at -336.4 MHz
gF_r = 0.1 # Ad hoc Lande factor to avoid

# Magnetic components |1> --> |2> --> |4> transition
mF_g = 2
Delta_g = mu_B_over_hbar * gF_g * mF_g
mF_e = 3
Delta_e = mu_B_over_hbar * gF_e * mF_e
mF_r = 2
Delta_r = mu_B_over_hbar * gF_r * mF_r

# Transmission Rydberg
sigma_24 = Sigma(Delta_c,Delta_p,Delta_g,Delta_e,Delta_r, Delta_s,Omega_c,Omega_p,Gamma_e,Gamma_r, gamma, phi, B)
#sigma_24 = SigmaPerturbative(Delta_c, Delta_p, Delta_g, Delta_e, Delta_r, Delta_s, Omega_c,Omega_p, Gamma_e, gamma, B)
#sigma_24 = SigmaPerturbativeAnalytic(Delta_c, Delta_p, Delta_g, Delta_e, Delta_r, Delta_s, Omega_c, Omega_p, Gamma_e, gamma)



# Magnetic components |1> --> |2> --> |4> transition
mF_g = -2
Delta_g = mu_B_over_hbar * gF_g * mF_g
mF_e = -3
Delta_e = mu_B_over_hbar * gF_e * mF_e
mF_r = -2
Delta_r = mu_B_over_hbar * gF_r * mF_r

# Transmission Rydberg
sigma_79 = Sigma(Delta_c,Delta_p,Delta_g,Delta_e,Delta_r, Delta_s,Omega_c,Omega_p,Gamma_e,Gamma_r, gamma, phi, B)
#sigma_79 = SigmaPerturbative(Delta_c, Delta_p, Delta_g, Delta_e, Delta_r, Delta_s, Omega_c,Omega_p, Gamma_e, gamma, B)
#sigma_79 = SigmaPerturbativeAnalytic(Delta_c, Delta_p, Delta_g, Delta_e, Delta_r, Delta_s, Omega_c, Omega_p, Gamma_e, gamma)



# Total transmission
# T1 = np.exp(-alpha*Gamma_e*(np.array(sigma_23)+np.array(sigma_24)+np.array(sigma_25))/Omega_p) #+ np.exp(-alpha*Gamma_e*np.array(sigma_24)/Omega_p) + np.exp(-alpha*Gamma_e*np.array(sigma_25)/Omega_p)
# T2 = np.exp(-alpha*Gamma_e*(np.array(sigma_78)+np.array(sigma_79)+np.array(sigma_710))/Omega_p) #+ np.exp(-alpha*Gamma_e*np.array(sigma_79)/Omega_p) + np.exp(-alpha*Gamma_e*np.array(sigma_710)/Omega_p)
# T = T1+T2
T = np.exp(-alpha*(np.array(sigma_23)*np.array(sigma_24)*np.array(sigma_25)*np.array(sigma_78)*np.array(sigma_79)*np.array(sigma_710)))
# Plot
plt.figure()
plt.plot(Delta_c, T*100)
plt.text(100, 27, r'$|33D_{5/2},mj=\pm5/2, mf=\pm4>$', fontsize=8)
plt.xlabel(r'$\Delta_c$/2$\pi$ [MHz]')
plt.ylabel(r'Transmission [%]')

plt.show()






