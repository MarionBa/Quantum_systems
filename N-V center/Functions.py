import numpy as np
import random


""" This script contains all the functions which are needed for running the Energy_states.py as well 
as the ODMR_spectrum.py scripts """

### Parameters ###
# In this section, I put parameters which I take from the literature and are needed for some
# of my OBE functions

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
    [1] H. El-Ella et al. “Optimised frequency modulation for continuous-wave
    optical magnetic resonance sensing using nitrogen-vacancy ensembles” Vol. 25, No. 13 OPTICS EXPRESS (2017)


    Parameters
    ----------
    the energy difference between the ground states : omega_0

    the excitation/relaxation rate due to the green laser : Gamma_p

    Omega : the Rabi frequency of the MW field

    omega_c : the tunned frequency of the MW field [array]

    gamma_2 : the dephasing term (decoherence quantum system due to environment)

    Returns
    -------
    the fluorescence ratio as derived in the reference paper [1]

    """

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



def Spectrum_NoSpinFlip_analytic(D_GS, Gamma_p, Omega, gamma_2, omega_c, Ro, Stot):

    """ Calculation of the fluorescence spectrum from the solution of the OBE as in the reference paper:
      [1] H. El-Ella et al. “Optimised frequency modulation for continuous-wave
      optical magnetic resonance sensing using nitrogen-vacancy ensembles” Vol. 25, No. 13 OPTICS EXPRESS (2017)

      Parameters
      ----------
      the energy difference between the ground states : D_GS

      the excitation/relaxation rate due to the green laser : Gamma_p

      Omega : the Rabi frequency of the MW field

      omega_c : the tunned frequency of the MW field [array]

      gamma_2 : the dephasing term (decoherence quantum system due to environment)

      Ro: the fluorescence baseline

      Stot: the total spin of the Nitrogen in the NV center, takes value 1 or 1/2 (Nitrogen 15 and 14)

      Returns
      -------
      the fluorescence ratio as derived in the reference paper [1]

      """

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



def OBE_3levels(omega_0, Gamma_p, Omega, omega_c, gamma_2):

    """ Solves the OBE for a NV center represented as a 3 level system ms=0,+1,-1

    Parameters
    ----------
    the energy difference between the ground states : omega_0

    the excitation/relaxation rate due to the green laser : Gamma_p

    Omega : the Rabi frequency of the MW field

    omega_c : the tunned frequency of the MW field [array]

    gamma_2 : the dephasing term (decoherence quantum system due to environment)

    Returns
    -------
    the density of state for ms=0 which is directly proportional to the fluorescence spectrum """

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

    # Corresponding vector
    Vector = [0, 0, 0, 0, 0, 0, 0, 0, 1]

    # Solving the linear system of equations
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

    """ Creates a 2D Gaussian profile

    Parameters
    ----------
    shape : (x, y)
        Output array shape (rows, cols).
    center : (xo, yo)
        Center of the gaussian in array coordinates (row, col).
    smearing : (sigma_x, sigma_y)
        The denominator in the exponential of the gaussian function
    Eo:
        Amplitude of the gaussian, here the electric field

    Returns
    -------
    mask : ndarray, shape (x,y), dtype uint8

    """

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

    ### Calculate the corresponding N+ electric field felt for a given green laser field felt ###

    ### Fundamental constants ### - Do not modify
    eo = 8.854 * 10 ** -12 # Permitivity vacuum [C^2.kg^-1.m^-3.s^2]
    c = 2.99 * 10 ** 8 # Speed of light [m.s^-1]
    n_d = 2.417 # Refractive index for green light in diamond

    # Core calculations
    I_NV = (E_NV ** 2 * c * n_d * eo) /2  # Laser field amplitude
    I = np.linspace(I_NV-I_NV/2, I_NV+I_NV/2, 1000)
    k0 = 2 * 10 ** -6 # (5.10^-7-2.10^6)
    Neutral = Er-k0*I # calculation of the effective N+ charge neutralization due to the green laser field I
    arg = np.argmin(abs(I-I_NV))
    E_eff = Neutral[arg] # Effective field derived

    return E_eff


def Gauss(x, A, B):
    """ Returns a Gaussian function """
    return A * np.exp(-B*x**2)








