import numpy as np
import Functions as fct
import matplotlib.pyplot as plt

""" Ref. paper doi:  DOI: 10.1038/NPHYS1969"""
""" This script calculates the eigenstates profile (smearing) of 
NV centers in a diamond sample under green laser illumination. 
 It generates a quantity which describes the energy states smearing and
 can be used to compute the ODMR spectrum. """

### Fundamental constants ### - Do not modify
eo = 8.854 * 10 ** -12 # Permitivity vacuum [C^2.kg^-1.m^-3.s^2]
c = 2.99 * 10 ** 8 # Speed of light [m.s^-1]
n_d = 2.417 # Refractive index for green light in diamond
D_GS = 2.87 * 10 ** 9 # Ground state bare energy N-V center [Hz]
lat = 0.3567 * 10 ** -9 # [m] diamond lattice spacing

# Directional electric susceptibility diamond with respect to the N-V axis
d_perp = 17 * 10 ** -2 # perpendicular susceptibility [Hz.m/V]
d_paral = 0.35 * 10 ** -2 # parallel susceptibility [Hz.m/V]


### Parameters ### - You can modify

I = 13 * 10 ** 6 # green laser intensity [W/m^2]
rho = 5000 # N-V center concentration in diamond sample [ppm]

# Laser beam spot dimensions, we assume an ellipse shape (if width = height --> disc)
d_spot_H = 0.1 * 10 ** -6 # Height ellipse [m]
d_spot_W = d_spot_H # Width ellipse [m]

###################################################################
###################################################################

""" Statistical calculation of electric field sensed by the N-V centers in our diamond sample.
1- We define a square/rectangular area with dimension of the laser beam spot
2- We randomly place NV centers in the square/rectangle area following to the NV center concentration
3- We place an ellipse/circle mask with Gaussian distribution on our area
--> We obtain randomly distributed N-V centers in a Gaussian laser beam spot with ellipse/circular profile"""

### Initialization ###

N = int(d_spot_H/lat)*int(d_spot_W/lat) # Number of diamond lattices in d_spot_H x d_spot_W square
N_NV = int(rho * N / 10 ** 6) # number of N-V centers in d_spot_H x d_spot_W square

# (x, y) max values --> correspond to number of diamond lattice in beam spot diameter
x_max = int(np.sqrt(N))
y_max = int(np.sqrt(N))

Eo = np.sqrt(2*I/(c*n_d*eo)) # Laser field amplitude


### Generate random N-V center in my sample ###
Random_NV = fct.randomNV(x_max, y_max, N_NV)

### Create ellipse mask and apply it on my sample ###
shape = x_max, y_max
center = x_max/2, y_max/2
axes = x_max/2, y_max/2
Ellipse_mask = fct.ellipse_mask_2d(shape, center, axes, int(np.sqrt(N)))
Ellipse_array = np.multiply(Ellipse_mask, Random_NV)

### Create Gaussian profile and apply it on my sample with ellipse mask ###
Gaussian = fct.twoD_Gaussian(np.arange(0, x_max), np.arange(0, y_max), Eo, center[0], center[1], center[0]/2, center[1]/2)
Gaussian_field_profile = np.multiply(Ellipse_array, Gaussian)

### Plotting figures of Gaussian profile with ellipse mask & N-V center distribution with corresponding field amplitude ###
plt.figure()
plt.pcolor(np.multiply(Gaussian, Ellipse_mask), edgecolors='none', linewidths=4)
cbar = plt.colorbar(orientation="vertical")
plt.title('Green laser beam spot profile')
cbar.set_label(r'$|E_0|$')

plt.figure(2)
plt.pcolor(Gaussian_field_profile, edgecolors='none', linewidths=4)
cbar = plt.colorbar(orientation="vertical")
plt.title(r'N-V centers distribution ' + str(rho) + 'ppm')
cbar.set_label(r'$|E_0|$')

plt.show()


###################################################################
###################################################################

""" Diagonalization of Hamiltonian for each NV center in my sample.
1- Select my NV center coordinates in my sample
2- Diagonalize Hamiltonian for each NV center in my sample 
3- Store eigenvalues """


# Selecting the NV centers in my sample and extracting their coordinates
nonzero_indices = np.nonzero(Gaussian_field_profile)
no_zero = Gaussian_field_profile[nonzero_indices]

# Initializing arrays
S0Ms0 = []
S1Msp1 = []
S1Msm1 = []
Field = []

# Loop over every single NV center
for E_NV in no_zero:

      # Save information about the electric field distribution
      Field.append(E_NV)

      ### Electric field initialization for NV center ###
      Ez = 0 # z direction polarization: 0 except in waveguides on chip!
      Ex = E_NV/2 # Polarization along x axis
      Ey = np.sqrt(np.square(E_NV)-np.square(Ex)) # Resulting polarization along y axis

      ### Definition of parameters Hamiltonian according to polarization field ###
      Pi_z = d_paral * Ez
      Pi_x = d_perp * Ex
      Pi_y = d_perp * Ey

      ### Hamiltonian to diagonalize - Basis ms = {-1, 0, 1} ###
      Hi = [[D_GS+Pi_z,              0,   -1j*Pi_x-Pi_y],
            [0,                      0,               0],
            [1j*Pi_x-Pi_y,           0,       D_GS+Pi_z]]


      ### Diagonalize Hamiltonian ###
      eigenvalues, eigenvectors = np.linalg.eigh(Hi)

      ### Save eigenvalues in arrays ###
      Factor = 10 ** -6 # Factor for expressing eigenstates in MHz
      S0Ms0.append(eigenvalues[0]*Factor)
      S1Msm1.append(eigenvalues[1]*Factor)
      S1Msp1.append(eigenvalues[2]*Factor)


###################################################################
###################################################################

""" Creating visual on energy smearing of eigenstates """

# Initializing frequency arrays
df = 10 ** -3
Frequencies_ms0 = np.arange(-0.5 * 10 ** -2, max(S0Ms0)+0.5 * 10 ** -2, df)
Frequencies_ms1 = np.arange(min(S1Msm1)-0.5 * 10 ** -1, max(S1Msp1)+0.5 * 10 ** -1, df)

# Initializing visual eigenstates arrays
Eigenfrequencies_ms0 = np.ones(len(Frequencies_ms0))
Eigenfrequencies_ms1 = np.ones(len(Frequencies_ms1))

# Replacing values of visual eigenstates arrays with weighted (variance/mean value) eigenstates frequencies
for i in range(len(S0Ms0)):

      # State |S=1 ms = 0>
      idx = np.argmin(abs(S0Ms0[i]-Frequencies_ms0))
      Eigenfrequencies_ms0[idx] = 1 - abs(abs(S0Ms0[i] - np.mean(S0Ms0)) / np.mean(S0Ms0))

      # State |S=1 ms = +1>
      idx = np.argmin(abs(S1Msp1[i] - Frequencies_ms1))
      Eigenfrequencies_ms1[idx] = 1 - abs(abs(S1Msp1[i] - np.mean(S1Msp1)) / np.mean(S1Msp1)) * 10 ** 6

      # State |S=1 ms = -1>
      idx = np.argmin(abs(S1Msm1[i] - Frequencies_ms1))
      Eigenfrequencies_ms1[idx] = 1 - abs(abs(S1Msm1[i] - np.mean(S1Msm1)) / np.mean(S1Msm1)) * 10 ** 6


# Figures of the S=0 and S=1 eigenstates visualization
plt.figure()
plt.plot(Frequencies_ms0, Eigenfrequencies_ms0)
plt.gca().set_yticks([])
plt.xlabel('Frequency [MHz]')
plt.title(r'GS relative frequency, $m_s=0$')

plt.figure()
plt.plot(Frequencies_ms1, Eigenfrequencies_ms1)
plt.gca().set_yticks([])
xticks = np.arange(min(S1Msm1)-0.5 * 10 ** -1, max(S1Msp1)+0.5 * 10 ** -1, 0.05)
xlabels = [f'{x:1.2f}' for x in xticks]
plt.xticks(xticks, labels=xlabels)
plt.xlabel('Frequency [MHz]')
plt.title(r'Eigenstates relative frequency with respect to GS, $m_s=\pm 1$')
plt.show()


# Figure to visualize the discrete eigenstates
# fig, ax = plt.subplots(figsize=(6, 8))
# Factor = 10 ** -6
# ax.plot([0, 1], [eigenvalues[0] * Factor, eigenvalues[0] * Factor], color='tab:blue', linewidth=4)
# ax.plot([0, 1], [eigenvalues[1] * Factor, eigenvalues[1] * Factor], color='tab:blue', linewidth=4)
# ax.plot([0, 1], [eigenvalues[2] * Factor, eigenvalues[2] * Factor], color='tab:blue', linewidth=4)
#
# ax.set_ylabel("Relative ground states [MHz]")
#
# plt.text(0.8, 3000, r'$|S=1,m_s=1>$', fontsize=15, color='tab:blue')
# plt.text(0.8, 100, r'$|S=1,m_s=0>$', fontsize=15, color='tab:blue')
#
# # Hide specific spines
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.set_xticks([])
#
# plt.show()