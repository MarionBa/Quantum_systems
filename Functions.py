import numpy as np

""" Script for the computation of components of the absorption cross section \Sigma_ij as in 
the reference paper H-J Su et al. "Optimizing the Rydberg EIT spectrum in a thermal vapor" Optics Express 30,2 (2022) """

# Parameters system
lambda_p = 780 * 10 ** -9  # Wavelength probe [nm]
lambda_c = 480 * 10 ** -9  # Wavelength coupling [nm]


""" \Sigma_ij obtained with the analytical equation Eq. 6 """

def SigmaPerturbativeAnalytic(Delta_c,Delta_p, Delta_s, Delta_g, Delta_e, Delta_r, Omega_c, Gamma_e, gamma, B):
    sigma = []
    for elt in Delta_c:
        # Coupling laser detuning
        Delta_c_idx = elt * (2 * np.pi)

        # Doppler shift
        Delta_D = (Delta_e - Delta_g) * (lambda_p - lambda_c) / lambda_c
        delta = B * (Delta_g + Delta_e + Delta_r + Delta_D) + Delta_s + Delta_c_idx  # Two-photons detuning

        # The analytical equation Eq. 6
        sigma.append((Gamma_e ** 2 * (4 * gamma ** 2 + 4 * delta ** 2) + 2 * gamma * Omega_c ** 2 * Gamma_e) / (
                (2 * gamma * Gamma_e + Omega_c ** 2 - 4 * Delta_p * delta) ** 2 + (
                    4 * Delta_p * gamma + 2 * Gamma_e * delta) ** 2))

    # Returns \sigma_ij
    return sigma


""" \Sigma_ij obtained by numerically solving Eqs. 4a-b """

def SigmaPerturbative(Delta_c,Delta_p, Delta_g, Delta_e, Delta_r, Delta_s,Omega_c, Omega_p,Gamma_e, gamma, B):
    Im_rho_eg = []
    for elt in Delta_c:
        # Coupling laser detuning
        Delta_c_idx = elt * (2 * np.pi)

        # Doppler shift
        Delta_D = (Delta_e - Delta_g) * (lambda_p - lambda_c) / lambda_c
        delta = B * (Delta_g + Delta_e + Delta_r + Delta_D) + Delta_s + Delta_c_idx  # Two-photons detuning

        # Square matrix to be solved with corresponding vector, Eqs. 4a-b in steady state
        # Using matrix inversion method
        Matrix = [[1j*Omega_c/2, -(Gamma_e/2+1j*Delta_p)],
                  [-(gamma+1j*delta), 1j*Omega_c/2]]
        Vector = [-1j*Omega_p/2, 0]
        Solution = np.linalg.solve(Matrix, Vector)

        # Taking the imaginary part of \rho_ij
        Im_rho_eg.append(np.imag((Solution[1]-np.conjugate(Solution[1]))/2))

    # Returns \sigma_ij
    return Gamma_e * np.array(Im_rho_eg) / Omega_p


""" \Sigma_ij obtained by numerically solving Eqs. 3a-f """

def Sigma(Delta_c,Delta_p, Delta_s, Delta_g, Delta_e, Delta_r, Omega_c, Omega_p, Gamma_e, Gamma_r, gamma, phi, B):
    Im_rho_eg = []
    for elt in Delta_c:
        # Coupling laser detuning
        Delta_c_idx = elt * (2 * np.pi)

        # Doppler shift
        Delta_D = (Delta_e - Delta_g) * (lambda_p - lambda_c) / lambda_c
        delta = B * (Delta_g + Delta_e + Delta_r + Delta_D) + Delta_s + Delta_c_idx  # Two-photons detuning

        # Matrix to be solved with corresponding vector, Eqs. 4a-b in steady state
        # Chose to not go for square matrix because my problem was underdefined with less
        # equations to solve
        # Basis = [Density, \rho_gg, \rho_ee, \rho_rr, \rho_gg, \rho_eg, \rho_eg*, \rho_er, \rho_er*, \rho_gr, \rho_gr*]
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
        Vector = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # Basis for the vector with solutions [\rho_gg, \rho_ee, \rho_rr, \rho_gg, \rho_eg, \rho_eg*, \rho_er, \rho_er*, \rho_gr, \rho_gr*]
        Solution = np.linalg.lstsq(Matrix, Vector)
        Sol = Solution[0]

        # Taking the imaginary part of \rho_ij
        Im_rho_eg.append(np.imag((Sol[3]-Sol[4])/2))

    # Return \sigma_ij
    return Gamma_e * np.array(Im_rho_eg) / Omega_p