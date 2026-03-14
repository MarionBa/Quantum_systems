import numpy as np
import Functions as fct
import matplotlib.pyplot as plt
import sys
import random

""" Ref. paper doi:  DOI: 10.1038/NPHYS1969 """

""" 

This script calculates the eigenstates profile (smearing) of 
NV centers in a diamond sample under green laser illumination. 
 It generates a quantity which describes the energy states smearing and
 can be used to compute the ODMR spectrum. 
 
 Please scroll down the code and select the type of NV centers illumination method:
 1- Confocal microscopy
 2- Evanescent field
 
 """

dir = r"C:\Users\barbea43\OneDrive - imec\Documents\Picsys\N-V center\Results"

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

I = 1 * 10 ** 4 # green laser intensity [W/m^2]

# If bulk diamond sample
rho = 1000 # N-V center concentration in diamond sample [ppm]

# If nano diamonds
N_d = 10 # Number of nano-diamonds
rho_nano = 500 # N-V center concentration in diamond sample [ppm]
d_d = 0.05 * 10 ** -6 # Diameter nano diamond [m]

# Parameters for evanescent field
n_eff = 1.6 # Effective refractive index of the mode propagating along the WG
n_medium = 1 # Refractive index of the medium where the evanescent field (does not) propagate
lamb = 532 * 10 ** -9 # Wavelength of the green laser
ko = 2 * np.pi / lamb # Wavevector of the green laser

# Save parameters
np.save(dir + fr'\Energy splitting arrays\Parameters_Intensity{I}_Density{rho}', [I, rho])



###################################################################
###################################################################



""" 

In this section we simulate the illuminated diamond sample with NV centers, we propose
two different systems:

1- Confocal microscopy with a bulk diamond sample 'CM'
2- Evanescent field (from WG) with nano-diamonds 'EV'


"""


### Pick a system! ###
CM = False
EV = True

if CM == True:

      """ Statistical calculation of electric field sensed by the N-V centers in our diamond sample.
      1- We define a square/rectangular area with dimension of the laser beam spot
      2- We randomly place NV centers in the square/rectangle area following to the NV center concentration
      3- We place an ellipse/circle mask with Gaussian distribution on our area
      --> We obtain randomly distributed N-V centers in a Gaussian laser beam spot with ellipse/circular profile"""

      ### Initialization ###

      # Laser beam spot dimensions, we assume an ellipse shape (if width = height --> disc)
      d_spot_H = 0.1 * 10 ** -6  # Height ellipse [m]
      d_spot_W = 0.1 * 10 ** -6  # Width ellipse [m]

      N = int(d_spot_H / lat) * int(d_spot_W / lat)  # Number of diamond lattices in d_spot_H x d_spot_W square
      N_NV = int(rho * N / 10 ** 6)  # number of N-V centers in d_spot_H x d_spot_W square

      # Debugging
      if N_NV == 0:
            print('Need higher concentration of NV centers or bigger laser beam spot!')
            sys.exit()

      # (x, y) max values --> correspond to number of diamond lattice in beam spot diameter
      x_max = max(int(d_spot_H / lat), int(d_spot_W / lat))
      y_max = max(int(d_spot_H / lat), int(d_spot_W / lat))

      Eo = np.sqrt(2 * I / (c * n_d * eo))  # Laser field amplitude

      ### Generate random N-V center in my sample ###
      Random_NV = fct.randomNV(x_max, y_max, N_NV)

      ### Create ellipse mask and apply it on my sample ###
      shape = x_max, y_max
      center = x_max/2, y_max/2
      axes = int(d_spot_H/lat)/2, int(d_spot_W/lat)/2
      Ellipse_mask = fct.ellipse_mask_2d(shape, center, axes)
      Ellipse_array = np.multiply(Ellipse_mask, Random_NV)


      ### Create Gaussian profile and apply it on my sample with ellipse mask ###
      Gaussian = fct.twoD_Gaussian(np.arange(0, x_max), np.arange(0, y_max), Eo, center[0], center[1], center[0]/2, center[1]/2)
      Field_profile = np.multiply(Ellipse_array, Gaussian)


      ### Plotting figures of Gaussian profile with ellipse mask & N-V center distribution with corresponding field amplitude ###
      plt.figure()
      plt.pcolor(np.multiply(Gaussian, Ellipse_mask), edgecolors='none', linewidths=4)
      cbar = plt.colorbar(orientation="vertical")
      plt.title('Green laser beam spot profile')
      plt.xlabel(r'Diamond lattice cell $[0.35nm]$')
      plt.ylabel(r'Diamond lattice cell $[0.35nm]$')
      cbar.set_label(r'$|E_0|$')
      plt.savefig(dir + fr'\Field_distribution_Intensity{I}_Density{rho}_SpotH{d_spot_H}_SpotW{d_spot_W}')

      plt.figure()
      plt.pcolor(Field_profile, edgecolors='none', linewidths=4)
      cbar = plt.colorbar(orientation="vertical")
      plt.title(r'N-V centers distribution ' + str(rho) + 'ppm')
      cbar.set_label(r'$|E_0|$')
      plt.xlabel(r'Diamond lattice cell $[0.35nm]$')
      plt.ylabel(r'Diamond lattice cell $[0.35nm]$')
      plt.savefig(dir + fr'\NV_distribution_Intensity{I}_Density{rho}_SpotH{d_spot_H}_SpotW{d_spot_W}')

      plt.show()

elif EV == True:

      """ Statistical calculation of electric field sensed by the N-V centers in our nano diamonds.
            1- We define a square/rectangular area with dimension of the laser beam spot
            2- We randomly place diamonds in the square/rectangle area following to the diamond concentration
            
            --> We obtain randomly distributed N-V centers in an evanescent field from a WG"""

      ### Initialization ###

      # WG surface which generates evanescent field
      d_spot_H = 0.5 * 10 ** -6  # Length WG [m]
      d_spot_W = 0.15 * 10 ** -6  # Width WG [m]

      N = int(d_d / lat) * int(d_d / lat)  # Number of diamond lattices in a nano diamond - assumed to be squared nano diamonds
      N_nano = int(rho * N / 10 ** 6) * N_d


      # (x, y) max values --> correspond to number of diamond lattice in beam spot diameter
      x_max = int(d_spot_W / d_d)
      y_max = int(d_spot_H / d_d)
      z_max = int(d_d / lat)  # Maximum distance from the WG covered by the nano diamond


      ### Generate random nano diamonds positions in my sample ###
      Random_d = fct.randomNV_3D(x_max, y_max, z_max, N_nano)


      ### Create an evanescent field mask and apply it on my sample ###
      delta = 1 / (ko * np.sqrt(n_eff ** 2 - n_medium ** 2)) # Evanescent field depth
      shape = x_max, y_max, z_max

      Eo = np.sqrt(2 * I / (c * n_d * eo))  # Laser field amplitude

      Evanescent_mask = fct.evanescent_mask_3d(Eo, shape, delta, lat)
      Field_profile = np.multiply(Evanescent_mask, Random_d)

      # Check plots
      fig = plt.figure()
      ax = plt.axes(projection='3d')
      z, x, y = np.nonzero(Field_profile)
      color_values = I * np.exp(-z*lat/delta)
      p = ax.scatter(z*lat*10**6, x*d_d*10**6, y*d_d*10**6, c=color_values, edgecolors='none', linewidths=4)
      fig.colorbar(p, label=r'Intensity laser felt by NV centers [W/$m^2$]')
      ax.set_yticks(np.arange(0, 0.1, 0.05))
      ax.grid(False)
      ax.view_init(270, 180, 180)
      ax.set_xlabel(r'Distance from waveguide [$\mu$m]')
      ax.set_ylabel(r'Waveguide width [$\mu$m]')
      ax.set_zlabel(r'Waveguide length [$\mu$m]')
      ax.set_box_aspect([2, 0.4, 1])


      plt.figure(figsize=(12, 5))
      z = np.linspace(0, 0.4*10**-6, 100)
      Intensity = I * np.exp(-z/delta)
      plt.plot(z*10**6, Intensity, linewidth=4)
      plt.xlabel(r'Distance from waveguide [$\mu$m]')
      plt.ylabel(r'Evanescent intensity profile [W/$m^2$]')
      plt.vlines(-0.01, min(Intensity), max(Intensity), linewidth=25, colors='lightblue')
      plt.ylim(min(Intensity), max(Intensity))
      plt.xlim(-0.02, max(z*10**6))
      plt.show()


elif (EV == True) & (CM == True):
      print('You cannot have a system with both confocal microscopy and evanescent field')
elif (EV == False) & (CM == False):
      print('Pick a system; confocal microscopy (CM) or evanescent field (EV)')

###################################################################
###################################################################

""" Diagonalization of Hamiltonian for each NV center in my sample.
1- Select my NV center coordinates in my sample
2- Diagonalize Hamiltonian for each NV center in my sample 
3- Store eigenvalues """


# Selecting the NV centers in my sample and extracting their coordinates
nonzero_indices = np.nonzero(Field_profile)
no_zero = Field_profile[nonzero_indices]

# Initializing arrays
S0Ms0 = []
S1Msp1 = []
S1Msm1 = []
Field = []
DeltaE1 = []
DeltaE2 = []

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
      DeltaE1.append(abs(eigenvalues[0]-eigenvalues[1])*Factor)
      DeltaE2.append(abs(eigenvalues[1]-eigenvalues[2])*Factor)

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


########################################################################################################

# Save the states energy splitting array
np.random.shuffle(DeltaE1)
np.save(dir + fr'\Energy splitting arrays\Energy_states_splitting_Intensity{I}_Density{rho}_SpotH{d_spot_H}_SpotW{d_spot_W}', DeltaE1)

### Figures ###

# Figures of the S=0 and S=1 eigenstates energy splitting visualization with respect to its mean value
plt.figure()
Number_NV = np.arange(0, len(no_zero))
plt.scatter(Number_NV, (np.array(DeltaE1)-np.mean(DeltaE1))*10**3)
plt.xlabel('NV center index')
plt.ylabel('Relative spin states splitting [kHz]')
plt.title(r'Spin states splitting per NV center')
#plt.savefig(dir + fr'\Energy_states_splitting_Intensity{I}_Density{rho}_SpotH{d_spot_H}_SpotW{d_spot_W}')
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