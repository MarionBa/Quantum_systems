import numpy as np
import random


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

def Fluorescence(omega_0, Gamma_p, Omega, gamma_2, omega_c):

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

def Spectrum_NoSpinFlip_analytic(D_GS, Gamma_p, Omega, gamma_2, omega_c, Ro, Stot):

    # GS frequency
    omega_0 = D_GS

    # Detuning
    delta = omega_c - omega_0

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

def evanescent_mask_3d(Eo, shape, delta, lat):

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
    Array = np.ones((W, H))

    # Evanescent field equation: Eo * exp(-z/delta)
    inside = []
    for z in range(D):
        z_m = z * lat
        inside.append(Array * Eo * np.exp(-z_m/delta))

    return inside


