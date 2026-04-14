import numpy as np
import random
import matplotlib.pyplot as plt


""" Function which solves the OBE for N-V center and returns density of states \rho_33 and \rho_44 """

# def Spectrum_NoSpinFlip(Gamma_p, Omega, gamma_2, omega_c, Ro, Stot, Bz):
#
#     ### Parameters ###
#
#     # Gyromagnetic ratio
#     gamma_e = 28 # [MHz/mT]
#
#     # for spin-conserving transitions (non spin-flip)
#     k41 = 0
#     k32 = 0
#     k35 = 0
#     k52 = 0.2  # [MHz]
#     k51 = 0.32  # [MHz]
#     k31 = 77  # [MHz]
#     k45 = 66  # [MHz]
#     k42 = k31
#     # Values taken from L. Robledo et al. New J. Phys. 13 025013 (2011)
#
#     # spin-lattice relaxation k21<<1
#     k21 = 0
#
#     # Energy ground state
#     D_GS = 2870 # [MHz]
#     omega_0 = 2*np.pi * D_GS #( D_GS + HF + gamma_e * Bz)
#     delta = omega_c-omega_0
#     gamma_2_p = k21/2+gamma_2+Gamma_p/2
#
#     # Square matrix to be solved with corresponding vector, Eqs. 2 in steady state
#     # Using matrix inversion method
#     # Basis = {rho_11, rho_22, rho_33, rho_44, rho_55, rho_12, rho_21}
#     Matrix = [[-Gamma_p-k21/2, k21/2, k31, k41, k51, -1j*Omega/2, 1j*Omega/2],
#               [k21/2, -Gamma_p-k21/2, k32, k42, k52, 1j*Omega/2, -1j*Omega/2],
#               [Gamma_p, 0, -(k35+k32+k31), 0, 0, 0, 0],
#               [0, Gamma_p, 0, -(k45+k42+k41), 0, 0, 0],
#               [0, 0, k35, k45, -(k52+k51), 0, 0],
#               [-1j*Omega/2, 1j*Omega/2, 0, 0, 0, -(gamma_2_p-1j*delta), 0],
#               [1j*Omega/2, -1j*Omega/2, 0, 0, 0, 0, -(gamma_2_p+1j*delta)]]
#     Vector = [0, 0, 0, 0, 0, 0, 0]
#     #Solution = np.linalg.solve(Matrix, Vector)
#     Solution = np.linalg.lstsq(Matrix, Vector)
#     Sol = Solution[0]
#
#     # Density of states
#     rho_33 = Sol[2]
#     rho_44 = Sol[3]
#     print(Solution, Sol)
#     # Calculation of the fluorescence ratio
#     Icw = ((k31 + k32) * rho_33) / (k31 + k32 + k35) + ((k41 + k42) * rho_44) / (k41 + k42 + k45)
#
#     # Sepctrum calculation
#     if Stot == 1 : # 14N case
#         A = 2.16 # [MHz]
#         S = Ro * Icw * ((delta + np.pi * A) + delta + (delta - np.pi * A))
#     elif Stot == 1/2: # 15N case
#         A = 3.03 # [MHz]
#         S = Ro * Icw * ((delta + np.pi * A / 2) + (delta - np.pi * A / 2))
#     else:
#         print(r'Wrong $S_{tot}$ value.')
#
#     # Returns spectrum
#     return S

### Parameters ###

# Gyromagnetic ratio
gamma_e = 28 # [MHz/mT]

# for spin-conserving transitions (non spin-flip)
k41 = 0
k32 = 0
k35 = 0
k52 = 2 * np.pi * 0.2  # [MHz]
k51 = 2 * np.pi * 0.32  # [MHz]
k31 = 2 * np.pi * 77  # [MHz]
k45 = 2 * np.pi * 66  # [MHz]
k42 = k31
# Values taken from L. Robledo et al. New J. Phys. 13 025013 (2011)

# spin-lattice relaxation k21<<1
k21 = 0

def Fluorescence(omega_0, Gamma_p, Omega, gamma_2, omega_c):

    """ Calculation of the analytical equation solution of the OBE in
    H. El-Ella et al. “Optimised frequency modulation for continuous-wave
    optical magnetic resonance sensing using nitrogen-vacancy ensembles” Vol. 25, No. 13 OPTICS EXPRESS (2017)"""

    # Detunning
    delta = omega_c-omega_0

    # Dephasing
    gamma_2_p = k21/2+gamma_2+Gamma_p/2

    # Definitions
    K3 = k31 + k32 + k35
    K4 = k41 + k42 + k45
    K5 = k51 + k52

    # Ratio - Eq. (15)
    PP_num = k21/2 + Gamma_p*(k32*K5+k52*k35)/(K3*K5) + np.square(Omega)*gamma_2_p/(2*(np.square(gamma_2_p)+np.square(delta)))
    PP_den = Gamma_p + k21/2 - Gamma_p*(k42*K5+k52*k45)/(K4*K5) + np.square(Omega)*gamma_2_p/(2*(np.square(gamma_2_p)+np.square(delta)))
    PP = PP_num / PP_den

    # Calculation of the fluorescence ratio - Eq. (14)
    Icw_1stTerm = (Gamma_p*(k31+k32)/np.square(K3)) / (1 + PP + Gamma_p/K3 + Gamma_p*PP/K4 + k35*Gamma_p/(K3*K5) + k45*Gamma_p*PP/(K5*K4))
    Icw_2ndTerm = (Gamma_p*(k41+k42)/np.square(K4)) / (1 + 1/PP + Gamma_p/K4 + Gamma_p/(K3*PP) + k45*Gamma_p/(K4*K5) + k35*Gamma_p/(K5*K3*PP))
    return Icw_1stTerm + Icw_2ndTerm

def Fluorescence_2states(omega_0, omega_1, Gamma_p, Omega, delta, omega_c):

    """ Calculation of the OBE for a 2 levels NV center system """

    # MW frequency
    delta_c = omega_c-omega_0
    Es = omega_1-omega_0

    # Basis = {rho_11, rho_12, rho_21, rho_22}
    Matrix = [[1, 0, 0, 1],
              [Gamma_p, -1j*Omega/2, 1j*Omega/2, 0],
              [-1j*Omega, -1j*Es+delta+1j*delta_c, 0, 1j*Omega/2],
              [1j*Omega, 0, 1j*Es+delta-1j*delta_c, -1j*Omega],
              [-Gamma_p, 1j*Omega, -1j*Omega, 0]]

    Vector = [1, 0, 0, 0, 0]
    Solution = np.linalg.lstsq(Matrix, Vector)
    Sol = Solution[0]

    return np.real(Sol[3])



def Spectrum_NoSpinFlip_analytic(D_GS, Gamma_p, Omega, gamma_2, omega_c, Ro, Stot):

    """ Calculation of the fluorescence spectrum from the solution of the OBE
      H. El-Ella et al. “Optimised frequency modulation for continuous-wave
      optical magnetic resonance sensing using nitrogen-vacancy ensembles” Vol. 25, No. 13 OPTICS EXPRESS (2017)"""

    # GS frequency
    omega_0 = D_GS

    # Spectrum calculation
    if Stot == 1 : # 14N case
        # Hyperfine frequency
        A = 2.16 # [MHz]
        S = Ro * (Fluorescence(omega_0-2*np.pi*A, Gamma_p, Omega, gamma_2, omega_c) + Fluorescence(omega_0, Gamma_p, Omega, gamma_2, omega_c) + Fluorescence(omega_0+2*np.pi*A, Gamma_p, Omega, gamma_2, omega_c))

    elif Stot == 1/2: # 15N case
        # Hyperfine frequency
        A = 3.03 # [MHz]
        S = Ro * (Fluorescence(omega_0-np.pi*A, Gamma_p, Omega, gamma_2, omega_c) + Fluorescence(omega_0+np.pi*A, Gamma_p, Omega, gamma_2, omega_c))
    else:
        print(r'Wrong $S_{tot}$ value.')

    # Returns spectrum
    return S

def Spectrum_NoSpinFlip_analytic_2states(Gamma_p, Omega, delta, omega_c, omega_0, omega_1, Ro, Stot):

    """ Calculation of the fluorescence spectrum from the solution of the 2 levels OBE """

    # Spectrum calculation
    if Stot == 1: # 14N case
        # Hyperfine frequency
        A = 2.16 # [MHz]
        S = Ro * (Fluorescence_2states(omega_0-2*np.pi*A, omega_1, Gamma_p, Omega, delta, omega_c) + Fluorescence_2states(omega_0, omega_1, Gamma_p, Omega, delta, omega_c) + Fluorescence_2states(omega_0+2*np.pi*A, omega_1, Gamma_p, Omega, delta, omega_c))

    elif Stot == 1/2: # 15N case
        # Hyperfine frequency
        A = 3.03 # [MHz]
        S = Ro * (Fluorescence_2states(omega_0-np.pi*A, omega_1, Gamma_p, Omega, delta, omega_c) + Fluorescence_2states(omega_0+np.pi*A, omega_1, Gamma_p, Omega, delta, omega_c))
    else:
        print(r'Wrong $S_{tot}$ value.')


    # Returns spectrum
    return S


def OBE_3levels(omega_0, Gamma_p, Omega, omega_c, gamma_2):

    Delta = omega_c-omega_0

    # Basis = {rho_00, rho_+1+1, rho_0+1, rho_0-1, rho_+10, rho_-10, rho_+1-1, rho_-1+1}
    Matrix = [[-2*Gamma_p, Gamma_p, 1j*Omega/2, 0, -1j*Omega/2, 0, 0, 0],
              [0, 0, -1j*Omega/2, 0, 1j*Omega/2, 0, 0, 0],
              [1j*Omega/2, -1j*Omega/2, -(1j*Delta+Gamma_p+gamma_2), 0, 0, 0, 0, 0],
              [0, 0, 0, -(Gamma_p+gamma_2), 0, 0, -1j*Omega/2, 0],
              [-1j*Omega/2, 1j*Omega/2, 0, 0, 1j*Delta-Gamma_p-gamma_2/2, 0, 0, 0],
              [0, 0, 0, 0, 0, -(Gamma_p+gamma_2/2), 0, 1j*Omega/2],
              [0, 0, 0, -1j*Omega/2, 0, 0, 1j*Delta, 0],
              [0, 0, 0, 0, 0, 1j*Omega/2, 0, -1j*Delta],
              [1, 1, 0, 0, 0, 0, 0, 0]]

    Vector = [0, 0, 0, 0, 0, 0, 0, 0, 1]
    Solution = np.linalg.lstsq(Matrix, Vector)
    Sol = Solution[0]

    return np.real(Sol[0])



def abs_coef(concentration): # Concentration in ppm

    """ Calculation of absorbtion coefficient from results of ref.
    https://royalsocietypublishing.org/rsta/article/382/2265/20220314/112517/Absorption-and-birefringence-study-for-reduced
    for 700nm laser light with different N-V center concentration in diamond lattice """

    x = np.linspace(0, 500, 1000)
    Unit = 10 ** 2 # cm^-1 to m^-1 unit
    y = (np.square(x) / 11) * Unit
    idx = np.argmin(np.abs(x-concentration))
    return y[idx]

def twoD_Gaussian(x, y, Eo, xo, yo, sigma_x, sigma_y):

    """ Creates a 2D Gaussian profile """

    [X, Y] = np.meshgrid(x, y)
    g = np.exp(- (((X-xo)/sigma_x)**2 + (((Y-yo)/sigma_y)**2)))
    f = g.flatten()
    n = (f - f.min()) / (f.max() - f.min())
    g_norm = Eo * n.reshape(g.shape)
    return g_norm



def ellipse_mask_2d(shape, center, axes):
    """
    Create a 2D array of zeros with an axis-aligned ellipse of ones.

    Parameters
    ----------
    shape : (H, W)
        Output array shape (rows, cols).
    center : (cy, cx)
        Center of the ellipse in array coordinates (row, col).
    axes : (by, ax)
        Semi-axes (radii) along y and x, respectively.
    N:
        Number of points in my mesh

    Returns
    -------
    mask : ndarray, shape (H, W), dtype uint8
        1 inside ellipse, 0 outside.
    """

    W, H = shape
    cx, cy = center
    ax, by = axes

    # Coordinate grid
    x = np.linspace(0, W, W)
    y = np.linspace(0, H, H)
    [X, Y] = np.meshgrid(x, y)

    # Ellipse equation: ((x-cx)/ax)^2 + ((y-cy)/by)^2 <= 1
    inside = ((X - cx) / ax) ** 2 + ((Y - cy) / by) ** 2 <= 1.0

    return inside.astype(int)

def randomNV(x, y, N):
    """
        Generate random positions within 2D array.

        Parameters
        ----------
        x, y :
            Input array shape
        N :
            Number of NV center to be randomly positioned

        Returns
        -------
        2D array : ndarray, shape (x, y) of zeros with random ones

        """

    Init_array = np.zeros((x, y))
    for i in range(N):
          rand_x = random.randrange(0, x, 1)
          rand_y = random.randrange(0, y, 1)
          Init_array[rand_x, rand_y] = 1
    return Init_array

def randomNV_3D(x, y, z, N):
    """
        Generate random positions within 2D array.

        Parameters
        ----------
        x, y, z :
            Input array shape
        N :
            Number of NV center to be randomly positioned

        Returns
        -------
        3D array : ndarray, shape (x, y, z) of zeros with random ones

        """

    Init_array = np.zeros((z, x, y))
    for i in range(N):
          rand_x = random.randrange(0, x, 1)
          rand_y = random.randrange(0, y, 1)
          rand_z = random.randrange(0, z, 1)
          Init_array[rand_z, rand_x, rand_y] = 1
    return Init_array

def evanescent_mask_3d(I_loss, shape, delta, lat):

    """
    Create a 3D array of ones with an exponential decay along the z axis.

    Parameters
    ----------
    shape : (x, y, z)
        Output array shape (rows, cols, depth).

    Returns
    -------
    mask : ndarray, shape (x, y, z), dtype uint8
    """

    W, H, D = shape

    # Array initialization
    Array = np.ones((D, W, H))

    # Fundamental constants
    eo = 8.854 * 10 ** -12  # Permitivity vacuum [C^2.kg^-1.m^-3.s^2]
    c = 2.99 * 10 ** 8  # Speed of light [m.s^-1]
    n_d = 2.417  # Refractive index for green light in diamond

    # Evanescent field equation: Eo * exp(-z/delta)
    for z in range(D):
        for x in range(H):
            Eo = np.sqrt(2 * I_loss[x] / (c * n_d * eo))
            z_m = z * lat
            Array[z, :, x] = Eo * Array[z, :, x] * np.exp(-z_m/delta)

    return Array

def NV_centers_ionization(Er, E_NV):

    """"

    This function gives an effective value for the effect of the green laser intensity
    on the NV^- centers energy states.

    The green laser intensity indirectly influences the energy states of the NV^- center by
    mean of electric charge neutralization N^+ + NV^- --> NV^0 leading to a Stark effect
    on the ground states of the NV^- centers which is different for different green laser felt by the
    NV^- center.

    [1] Y.H. Yu et al. arXiv:2308.13351v2 [quant-ph] (2024)

    Assumptions:
    - Homogeneity of N^+, NV^0 and NV^- distributions
    - Local laser field felt by NV^- affects local DC electric field

    Input: Electric field felt by the NV center due to N^+

    Output: Effective electric field felt by the NV center due to N^+ variation as results of green laser field
    charge neutralization

    """


    ### Part 1: Behavior of the GS splitting as a function of the N concentration ###
    # Parameters from Fig. 3 (b) of [1]
    # N_conc_data = [100, 200, 300]
    # GS_split_data = [10, 17, 20]
    #
    # # Fitting of these data points
    # N_conc = np.linspace(0, 1000)
    # GS_split = 15*np.arctan(0.01*N_conc) # Function which fits best the data points [1]
    #
    # Checking the fitting of the arctan function with the data from [1]
    # plt.figure()
    # p = np.polyfit(N_conc_data, GS_split_data, deg=2)
    # plt.plot(N_conc, p[0]*N_conc**2 + p[1]*N_conc + p[2])
    # plt.plot(N_conc, GS_split)
    # plt.show()

    ### Part 2: Behavior of the charge neutralization rate with laser field intensity ###
    # From AI search of behavior in literature, we have 2 phases:
        # 1- The superlinear phase 0- ~100 uW/um^2 (~I^2)
        # 2- The linear phase >100 uW/um^2 (~I)

    # I_thresh = 100 * 10 ** 6 # [W/m^2] the field threshold for which the charge recombination regime changes
    # I_regime1 = np.linspace(10 ** -6, I_thresh, 1000) # [W/m^2]
    # I_regime2 = np.linspace(I_thresh, 10 ** 10, 1000) # [W/m^2]
    #
    # I = np.linspace(10, 10 ** 10, 10 ** 3)
    # Neutral = Er-(10**-7*I) # factor calculated with ionization cross sections
    # Checking the behavior of my charge neutralization profile
    # plt.figure()
    # plt.plot(I, E_eff)
    # plt.show()

    ### Part 3: Calculate the corresponding N+ electric field felt for a given green laser field felt ###

    ### Fundamental constants ### - Do not modify
    eo = 8.854 * 10 ** -12 # Permitivity vacuum [C^2.kg^-1.m^-3.s^2]
    c = 2.99 * 10 ** 8 # Speed of light [m.s^-1]
    n_d = 2.417 # Refractive index for green light in diamond

    I_NV = (E_NV ** 2 * c * n_d * eo) /2  # Laser field amplitude
    I = np.linspace(I_NV-I_NV/2, I_NV+I_NV/2, 1000)
    k0 = 2 * 10 ** -6 # (5.10^-7-2.10^6)
    Neutral = Er-k0*I
    arg = np.argmin(abs(I-I_NV))
    E_eff = Neutral[arg]

    return E_eff


def Gauss(x, A, B):
    """ Returns a Gaussian function """
    return A * np.exp(-B*x**2)








