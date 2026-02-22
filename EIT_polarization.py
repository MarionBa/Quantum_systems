import numpy as np
import Functions as fct
import matplotlib.pyplot as plt


""" Polarization of the coupling laser - Section 4.1 of the reference paper 
H-J Su et al. "Optimizing the Rydberg EIT spectrum in a thermal vapor" Optics Express 30,2 (2022) 
Note that this study only accounts for the population in the |1> state.
This code has variations to what is described in the reference paper:
1- We use an ad-hoc booster for the coupling Rabi frequency to fit the experimental/theoretical results, Eq.3, of the ref. Boost = 3
2- Use of sinusoid functions in order to vary the Clebcsh-Gordan coefficient as a function of the polarization
of the probe laser field --> fits nicely results of Fig. 2(b)
"""

### Saving directory, for figures ###
dir = r'C:\Users\barbea43\OneDrive - imec\Documents\Picsys\Rydberg atom\Results EIT peak optimization'

### General parameters of the system ###
Boost = 3
# Parameters in ref. paper for Fig. 3
Delta_c = np.linspace(-100, 500, 1000) # Coupling laser \Delta_c/2\pi [MHz]
alpha = 1.55 # Optical density [1]
Gamma_e = 2 * np.pi * 60 # Spontaneous emission excited state [MHz]
Gamma_r = 2 * np.pi * 5 # Spontaneous emission Rydberg state [MHz]
Omega_p = 2 * np.pi * 30 # The Rabi frequency of the probe laser [MHz]
Omega_c_CGC1 = Boost * 2 * np.pi * 5.2 # The Rabi frequency of the coupling laser for CGC = 1 [MHz]
gamma = 2 * np.pi * 8 # Coherence dephasing rate between |g> and |r>
Delta_p = 0 # Detuning probe [MHz]
# Magnetic parameters as in ref. paper
gF_g = 1/2 # lande factor ground state
gF_e = 2/3 # lande factor excited state
mu_B_over_hbar = 1 # [MHz/G]
B = 0


# Initialization arrays
T_array = []
Peak1_height = []
Peak2_height = []
Peak3_height = []

# Loop over i which will be used in defining the Clebsh Gordan coeficients
n = 0
i_step = 20
index = np.linspace(0, np.pi, i_step)
for i in index:
    n = n+1

    print(fr'Computing {(n-1)*100/(len(index)-1)}% done')


    ################################################################
    ################################################################

    #### Our 5/2 EIT peaks ####

    # Parameters for transitions to |r> states |3> and |8>
    CG_3 = 1 * 2/3 # Clebsh-Gordan coeficient and reduced dipole matrix ratio, from ref. paper
    Emp_fact = 1.5 # Empirical factor for fitting experimental data, from ref. paper
    Omega_c = CG_3 * np.sin(i) * Emp_fact * Omega_c_CGC1 # The Rabi frequency of the coupling laser [MHz]
    Delta_s = -2 * np.pi * 336.4 # EIT transition |33D_5/2>
    gF_r = 3 * 1.2 # Ad hoc Lande factor to control Zeeman splitting of the largest EIT peak

    ### |1> --> |2> --> |3> transition ###

    # Magnetic parameters
    mF_g = 2
    Delta_g = mu_B_over_hbar * gF_g * mF_g
    mF_e = 3
    Delta_e = mu_B_over_hbar * gF_e * mF_e
    mF_r = 4
    Delta_r = mu_B_over_hbar * gF_r * mF_r

    # Absorption cross-section
    sigma_23 = fct.SigmaPerturbative(Delta_c, Delta_p, Delta_g, Delta_e, Delta_r, Delta_s, Omega_c,Omega_p, Gamma_e, gamma, B)

    #######################################################################

    # Parameters for transitions to |r> states |4> and |9>
    CG_4 = 0.32 * 2/3 * np.cos(i) # Clebsh-Gordan coeficient and reduced dipole matrix ratio, from ref. paper
    Omega_c = CG_4 * Emp_fact * Omega_c_CGC1 # The Rabi frequency of the coupling laser [MHz]
    gF_r = 0.1 # Ad hoc Lande factor to avoid large Zeeman splitting and to fit ref. data better

    ### The |1> --> |2> --> |4> transition ###

    # Magnetic components
    mF_g = 2
    Delta_g = mu_B_over_hbar * gF_g * mF_g
    mF_e = 3
    Delta_e = mu_B_over_hbar * gF_e * mF_e
    mF_r = 2
    Delta_r = mu_B_over_hbar * gF_r * mF_r

    # Absorption cross-section
    sigma_24 = fct.SigmaPerturbative(Delta_c, Delta_p, Delta_g, Delta_e, Delta_r, Delta_s, Omega_c,Omega_p, Gamma_e, gamma, B)


    ################################################################
    ################################################################

    #### Our 3/2 EIT peak ####

    # Parameters for transitions to |r> states |5> and |10>
    CG_5 = 0.78 * 1/3 * np.cos(i) # Clebsh-Gordan coeficient and reduced dipole matrix ratio, from ref. paper
    Omega_c = CG_5 * Omega_c_CGC1 # The Rabi frequency of the coupling laser [MHz] (without empirical booster, as in ref.)
    Delta_s = 0 # EIT transition |33D_3/2>
    gF_r = 0.8

    ### The |1> --> |2> --> |5> transition ###

    # Magnetic components
    Delta_g = 0
    Delta_e = 0
    Delta_r = 0

    # Absorption cross-section
    sigma_25 = fct.SigmaPerturbative(Delta_c, Delta_p, Delta_g, Delta_e, Delta_r, Delta_s, Omega_c,Omega_p, Gamma_e, gamma, B)


    #######################################################################
    #######################################################################

    ### Total transmission - \sgima_eg = product of sigma_ij ###
    T = np.exp(-alpha*(np.array(sigma_23)*np.array(sigma_24)*np.array(sigma_25)))
    T_array.append(T)

    #######################################################################
    #######################################################################

    ### Extracting EIT peaks splitting ###

    # |1> --> |2> --> |3>
    T1 = np.exp(-alpha*(np.array(sigma_23)))
    Peak1_height.append((max(T1)-min(T1))*100)

    # |1> --> |2> --> |4>
    T2 = np.exp(-alpha * (np.array(sigma_24)))
    Peak2_height.append((max(T2)-min(T2))*100)

    # |1> --> |2> --> |5>
    T3 = np.exp(-alpha * (np.array(sigma_25)))
    Peak3_height.append((max(T3)-min(T3))*100)


    #######################################################################
    #######################################################################



### Figures ###

# Figure 1: spectra with the EIT peaks for different values of external magnetic field
plt.figure()
for i in range(0, len(index), 4):
    plt.plot(Delta_c, (T_array[i]-min(T_array[i]))*200 + i*int(max(index)), color='tab:blue')

# Editing yticks
listOf_Yticks = np.arange(-90, 91, 45)
plt.xticks(listOf_Yticks)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Labels
plt.xlabel(r'$\Delta_c$/2$\pi$ [MHz]', fontsize = 15)
plt.ylabel(r'Coupling $\lambda/4$ rotation angle ($^{\circ}$)', fontsize = 15)
plt.xlim([Delta_c[0], Delta_c[len(Delta_c)-1]])
plt.title(fr'Transmission spectra', fontsize = 15)
plt.savefig(dir + r'\Polarization\Spectra.png')

# Figure 2: peak heights
plt.figure(figsize=(8, 5))
Polarization = np.linspace(-90, 90, i_step)
plt.plot(Polarization, np.array(Peak1_height), '-o', color='k', linewidth=2)
plt.plot(Polarization, np.array(Peak2_height), '-o', color='b', linewidth=2)
plt.plot(Polarization, np.array(Peak3_height), '-o', color='g', linewidth=2)

# Editing xticks
listOf_Xticks = np.arange(-90, 91, 45)
plt.xticks(listOf_Xticks)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Labels
plt.ylabel(r'Peak height [%]', fontsize = 15)
plt.xlabel(r'Coupling $\lambda/4$ rotation angle ($^{\circ}$)', fontsize = 15)
plt.legend([r'$|1> \to |2> \to |3>$', r'$|1> \to |2> \to |4>$', r'$|1> \to |2> \to |5>$'])
plt.title(fr'Peak height as a function of coupling laser polarization', fontsize = 15)
plt.savefig(dir + r'\Polarization\EIT_peaks.png')

plt.show()



