import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths

# Transmission of Rydberg atom as function of coupling laser frequency


""" First figure without magnetic field and Doppler shift """
""" Transition |1> --> |2> --> |r>"""
""" Visual on all the peaks according to CGC for polarization sigma+ - sigma-"""
""" Ref. Su et al. Vol. 30, No.2/17 Jan. 2022 / Optics Express"""

Delta_c = np.linspace(-100, 500, 1000) # Coupling laser \Delta_c/2\pi [MHz]
alpha = 1.65 # Optical density [1]
Gamma_e = 2 * np.pi * 6 # Spontaneous emission excited state [MHz]
Omega_p = 2 * np.pi * 30 # The Rabi frequency of the probe laser [MHz]
gamma = 2 * np.pi * 5.4 # Coherence dephasing rate between |g> and |r>
Delta_p = 0 # Detuning probe --> we take resonant case


#######################################################################

# 3/2 state - |5>
CG_5 = 0.5 * 0.632 * 1/3 # Hyper fine, fine CGC and Wigner Eckart element respectively
Omega_c = CG_5 * 2 * np.pi * 7.8 # The Rabi frequency of the coupling laser [MHz]
Delta_s = 0 # EIT transition |33D_3/2>
delta = Delta_s + Delta_p + Delta_c * (2*np.pi) # Two-photons detuning


# Absorption cross-section
sigma_eg_5 = (Gamma_e ** 2 * (4 * gamma ** 2 + 4 * delta ** 2) + 2 * gamma * Omega_c ** 2 * Gamma_e) / ((2 * gamma * Gamma_e + Omega_c ** 2 - 4 * Delta_p * delta) ** 2 + (4 * Delta_p * gamma + 2 * Gamma_e * delta) ** 2)

#######################################################################

# 5/2 state - |4>
CG_4 = 0.189 * 0.316 * 1/3
Omega_c = CG_4 * 2 * np.pi * 7.8 # The Rabi frequency of the coupling laser [MHz]
Delta_s = -2 * np.pi * 336.4 # EIT transition |33D_5/2> --> needs minus sign otherwise symetric at -336.4 MHz
delta = Delta_s + Delta_p + Delta_c * (2*np.pi) # Two-photons detuning

# Absorption cross-section
sigma_eg_4 = (Gamma_e ** 2 * (4 * gamma ** 2 + 4 * delta ** 2) + 2 * gamma * Omega_c ** 2 * Gamma_e) / ((2 * gamma * Gamma_e + Omega_c ** 2 - 4 * Delta_p * delta) ** 2 + (4 * Delta_p * gamma + 2 * Gamma_e * delta) ** 2)

#######################################################################

# 5/2 state - |3>
CG_3 = 2*(1 * 1 * 1/3)/3 # Hyper fine, fine and WE coef.
Omega_c = CG_3 * 2 * np.pi * 7.8 # The Rabi frequency of the coupling laser [MHz]
Delta_s = -2 * np.pi * 336.4 # EIT transition |33D_5/2> --> needs minus sign otherwise symetric at -336.4 MHz
delta = Delta_s + Delta_p + Delta_c * (2*np.pi) # Two-photons detuning

# Absorption cross-section
sigma_eg_3 = (Gamma_e ** 2 * (4 * gamma ** 2 + 4 * delta ** 2) + 2 * gamma * Omega_c ** 2 * Gamma_e) / ((2 * gamma * Gamma_e + Omega_c ** 2 - 4 * Delta_p * delta) ** 2 + (4 * Delta_p * gamma + 2 * Gamma_e * delta) ** 2)


# Transmission Rydberg
T_3 = np.exp(-alpha*sigma_eg_3)
T_4 = np.exp(-alpha*sigma_eg_4)
T_5 = np.exp(-alpha*sigma_eg_5)
# Transition from |6>
T_8 = np.exp(-alpha*sigma_eg_3)
T_9 = np.exp(-alpha*sigma_eg_4)
T_10 = np.exp(-alpha*sigma_eg_5)
T_m = (T_5 + T_4 + T_3)/2 + (T_8 + T_9 + T_10)/2

# FWHM
peaks, _ = find_peaks(T_m)
results_half = peak_widths(T_m, peaks, rel_height=0.5)
FWHM = results_half[0]
print('Linewidth - polarization:', results_half[0])

# Plot
plt.figure(figsize=(8, 4))
plt.plot(Delta_c, T_m*100)
plt.text(200, 59.1, r'$|33D_{5/2},mj=\pm5/2, mf=2>$', fontsize=8)
plt.text(-10, 58, r'$|33D_{3/2},mj=\pm3/2, mf=2>$', fontsize=8)

plt.text(410, 58.5, 'FWHM '+str(format(FWHM[0], '.2f')), fontsize=8, color='tab:blue')
plt.arrow(275, 58.5, 50, 0, length_includes_head=True,
          head_width=0.04, head_length=5, color='tab:blue')
plt.arrow(400, 58.5, -50, 0, length_includes_head=True,
          head_width=0.04, head_length=5, color='tab:blue')


plt.xlabel(r'$\Delta_c$/2$\pi$ [MHz]')
plt.ylabel(r'Transmission [%]')
plt.xlim([Delta_c[0], Delta_c[len(Delta_c)-1]])
plt.title(r'Transmission spectrum $\sigma_-$ polarization')
plt.ylim([57.5, 61])

#######################################################################
#######################################################################
#######################################################################


""" First figure without magnetic field and Doppler shift """
""" Transition |1> --> |2> --> |r>"""
""" Visual on all the peaks according to CGC for polarization sigma+ """
""" Ref. Su et al. Vol. 30, No.2/17 Jan. 2022 / Optics Express"""

Delta_c = np.linspace(-100, 500, 1000) # Coupling laser \Delta_c/2\pi [MHz]
alpha = 1.65 # Optical density [1]
Gamma_e = 2 * np.pi * 6 # Spontaneous emission excited state [MHz]
Omega_p = 2 * np.pi * 30 # The Rabi frequency of the probe laser [MHz]
gamma = 2 * np.pi * 5.4 # Coherence dephasing rate between |g> and |r>
Delta_p = 0 # Detuning probe --> we take resonant case


#######################################################################

# 3/2 state - |5>
CG_5 = 0 #0.5 * 0.632 * 1/3 # Hyper fine, fine CGC and Wigner Eckart element respectively
Omega_c = CG_5 * 2 * np.pi * 7.8 # The Rabi frequency of the coupling laser [MHz]
Delta_s = 0 # EIT transition |33D_3/2>
delta = Delta_s + Delta_p + Delta_c * (2*np.pi) # Two-photons detuning


# Absorption cross-section
sigma_eg_5 = (Gamma_e ** 2 * (4 * gamma ** 2 + 4 * delta ** 2) + 2 * gamma * Omega_c ** 2 * Gamma_e) / ((2 * gamma * Gamma_e + Omega_c ** 2 - 4 * Delta_p * delta) ** 2 + (4 * Delta_p * gamma + 2 * Gamma_e * delta) ** 2)

#######################################################################

# 5/2 state - |4>
CG_4 = 0 #0.189 * 0.316 * 1/3
Omega_c = CG_4 * 2 * np.pi * 7.8 # The Rabi frequency of the coupling laser [MHz]
Delta_s = -2 * np.pi * 336.4 # EIT transition |33D_5/2> --> needs minus sign otherwise symetric at -336.4 MHz
delta = Delta_s + Delta_p + Delta_c * (2*np.pi) # Two-photons detuning

# Absorption cross-section
sigma_eg_4 = (Gamma_e ** 2 * (4 * gamma ** 2 + 4 * delta ** 2) + 2 * gamma * Omega_c ** 2 * Gamma_e) / ((2 * gamma * Gamma_e + Omega_c ** 2 - 4 * Delta_p * delta) ** 2 + (4 * Delta_p * gamma + 2 * Gamma_e * delta) ** 2)

#######################################################################

# 5/2 state - |3>
CG_3 = 1 * 1 * 1/3 # Hyper fine, fine and WE coef.
Omega_c = CG_3 * 2 * np.pi * 7.8 # The Rabi frequency of the coupling laser [MHz]
Delta_s = -2 * np.pi * 336.4 # EIT transition |33D_5/2> --> needs minus sign otherwise symetric at -336.4 MHz
delta = Delta_s + Delta_p + Delta_c * (2*np.pi) # Two-photons detuning

# Absorption cross-section
sigma_eg_3 = (Gamma_e ** 2 * (4 * gamma ** 2 + 4 * delta ** 2) + 2 * gamma * Omega_c ** 2 * Gamma_e) / ((2 * gamma * Gamma_e + Omega_c ** 2 - 4 * Delta_p * delta) ** 2 + (4 * Delta_p * gamma + 2 * Gamma_e * delta) ** 2)


# Transmission Rydberg
T_3 = np.exp(-alpha*sigma_eg_3)
T_4 = np.exp(-alpha*sigma_eg_4)
T_5 = np.exp(-alpha*sigma_eg_5)
# Transition from |6>
T_8 = np.exp(-alpha*sigma_eg_3)
T_9 = np.exp(-alpha*sigma_eg_4)
T_10 = np.exp(-alpha*sigma_eg_5)
T = (T_5 + T_4 + T_3)/2 + (T_8 + T_9 + T_10)/2

# FWHM
peaks, _ = find_peaks(T)
results_half = peak_widths(T, peaks, rel_height=0.5)
FWHM = results_half[0]
print('Linewidth + polarization:', results_half[0])

#Plot
plt.figure(figsize=(8, 4))
plt.plot(Delta_c, T*100, Delta_c, T_m*100, '--', color='tab:blue')
plt.text(150, 60.5, r'$|33D_{5/2},mj=\pm5/2, mf=3>$', fontsize=8)
plt.text(150, 59, r'$|33D_{5/2},mj=\pm5/2, mf=2>$', fontsize=8)
plt.text(-10, 58, r'$|33D_{3/2},mj=\pm3/2, mf=2>$', fontsize=8)

plt.text(410, 59.5, 'FWHM '+str(format(FWHM[0], '.2f')), fontsize=8, color='tab:blue')
plt.arrow(275, 59.5, 50, 0, length_includes_head=True,
          head_width=0.04, head_length=5, color='tab:blue')
plt.arrow(400, 59.5, -50, 0, length_includes_head=True,
          head_width=0.04, head_length=5, color='tab:blue')


plt.xlabel(r'$\Delta_c$/2$\pi$ [MHz]')
plt.ylabel(r'Transmission [%]')
plt.xlim([Delta_c[0], Delta_c[len(Delta_c)-1]])
plt.legend([r'$\sigma_+$', r'$\sigma_-$'])
plt.title(r'Transmission spectrum $\sigma_+$ polarization')
plt.ylim([57.5, 61])

#######################################################################
#######################################################################
#######################################################################


""" Looking at changes in probe frequency """
""" Transition |1> --> |2> --> |r>"""
""" Visual on all the peaks according to CGC for polarization sigma+ """
""" Ref. Su et al. Vol. 30, No.2/17 Jan. 2022 / Optics Express"""

Delta_c = np.linspace(100, 600, 1000) # Coupling laser \Delta_c/2\pi [MHz]
alpha = 1.65 # Optical density [1]
Gamma_e = 2 * np.pi * 6 # Spontaneous emission excited state [MHz]
Omega_p = 2 * np.pi * 30 # The Rabi frequency of the probe laser [MHz]
gamma = 2 * np.pi * 5.4 # Coherence dephasing rate between |g> and |r>


Transmission_Delta_p = []
FWHM_Delta_p = []
Peaks = []
Wave = []
Delta_max = 1
for Delta_p in np.linspace(-Delta_max, Delta_max, 10):
    print(Delta_p)
    Wave.append(Delta_p)
    #######################################################################

    # 3/2 state - |5>
    CG_5 = 0 #0.5 * 0.632 * 1/3 # Hyper fine, fine CGC and Wigner Eckart element respectively
    Omega_c = CG_5 * 2 * np.pi * 7.8 # The Rabi frequency of the coupling laser [MHz]
    Delta_s = 0 # EIT transition |33D_3/2>
    delta = Delta_s + Delta_p + Delta_c * (2*np.pi) # Two-photons detuning


    # Absorption cross-section
    sigma_eg_5 = (Gamma_e ** 2 * (4 * gamma ** 2 + 4 * delta ** 2) + 2 * gamma * Omega_c ** 2 * Gamma_e) / ((2 * gamma * Gamma_e + Omega_c ** 2 - 4 * Delta_p * delta) ** 2 + (4 * Delta_p * gamma + 2 * Gamma_e * delta) ** 2)

    #######################################################################

    # 5/2 state - |4>
    CG_4 = 0 #0.189 * 0.316 * 1/3
    Omega_c = CG_4 * 2 * np.pi * 7.8 # The Rabi frequency of the coupling laser [MHz]
    Delta_s = -2 * np.pi * 336.4 # EIT transition |33D_5/2> --> needs minus sign otherwise symetric at -336.4 MHz
    delta = Delta_s + Delta_p + Delta_c * (2*np.pi) # Two-photons detuning

    # Absorption cross-section
    sigma_eg_4 = (Gamma_e ** 2 * (4 * gamma ** 2 + 4 * delta ** 2) + 2 * gamma * Omega_c ** 2 * Gamma_e) / ((2 * gamma * Gamma_e + Omega_c ** 2 - 4 * Delta_p * delta) ** 2 + (4 * Delta_p * gamma + 2 * Gamma_e * delta) ** 2)

    #######################################################################

    # 5/2 state - |3>
    CG_3 = 1 * 1 * 1/3 # Hyper fine, fine and WE coef.
    Omega_c = CG_3 * 2 * np.pi * 7.8 # The Rabi frequency of the coupling laser [MHz]
    Delta_s = -2 * np.pi * 336.4 # EIT transition |33D_5/2> --> needs minus sign otherwise symetric at -336.4 MHz
    delta = Delta_s + Delta_p + Delta_c * (2*np.pi) # Two-photons detuning

    # Absorption cross-section
    sigma_eg_3 = (Gamma_e ** 2 * (4 * gamma ** 2 + 4 * delta ** 2) + 2 * gamma * Omega_c ** 2 * Gamma_e) / ((2 * gamma * Gamma_e + Omega_c ** 2 - 4 * Delta_p * delta) ** 2 + (4 * Delta_p * gamma + 2 * Gamma_e * delta) ** 2)


    # Transmission Rydberg
    T_3 = np.exp(-alpha*sigma_eg_3)
    T_4 = np.exp(-alpha*sigma_eg_4)
    T_5 = np.exp(-alpha*sigma_eg_5)
    # Transition from |6>
    T_8 = np.exp(-alpha*sigma_eg_3)
    T_9 = np.exp(-alpha*sigma_eg_4)
    T_10 = np.exp(-alpha*sigma_eg_5)
    Tp = (T_5 + T_4 + T_3)/2 + (T_8 + T_9 + T_10)/2
    Transmission_Delta_p.append(Tp)

    # FWHM
    peaks, _ = find_peaks(Tp)
    results_half = peak_widths(Tp, peaks, rel_height=0.5)
    FWHM = results_half[0]
    Peaks.append((max(Tp)-min(Tp))*100) #-min(Tp))
    FWHM_Delta_p.append(FWHM)



# Plot
# plt.figure(figsize=(8, 4))
# Wave =np.arange(-2, 3, 1)
# plt.plot(Delta_c, (Transmission_Delta_p[0]/(min(Transmission_Delta_p[0])-0*min(Transmission_Delta_p[0])))*100)
# plt.plot(Delta_c, (Transmission_Delta_p[1]/(min(Transmission_Delta_p[1])-0*min(Transmission_Delta_p[1])))*100)
# plt.plot(Delta_c, (Transmission_Delta_p[2]/(min(Transmission_Delta_p[2])-0*min(Transmission_Delta_p[2])))*100)
# plt.plot(Delta_c, (Transmission_Delta_p[3]/(min(Transmission_Delta_p[3])-0*min(Transmission_Delta_p[3])))*100)
# plt.plot(Delta_c, (Transmission_Delta_p[4]/(min(Transmission_Delta_p[4])-0*min(Transmission_Delta_p[4])))*100)
# plt.legend([str(Wave[0]) + 'MHz', str(Wave[1]) + 'MHz', str(Wave[2]) + 'MHz', str(Wave[3]) + 'MHz', str(Wave[4]) + 'MHz'])
# plt.text(150, 60.5, r'$|33D_{5/2},mj=\pm5/2, mf=3>$', fontsize=8)
# plt.xlabel(r'$\Delta_c$/2$\pi$ [MHz]')
# plt.ylabel(r'Transmission [arb.]')
# plt.xlim([Delta_c[0], Delta_c[len(Delta_c)-1]])
# plt.title(r'Influence of probe detuning on largest EIT peak')
# #plt.ylim([57.5, 61])


plt.figure()
plt.scatter(np.linspace(-Delta_max, Delta_max, 10), Peaks)
plt.xlabel('Probe frequency detuning [MHz]')
plt.ylabel('Transmission EIT peak amplitude [%]')
plt.title('EIT transmission as a function of the probe detuning')
plt.show()

#######################################################################
#######################################################################
#######################################################################


""" EIT for varying probe intensity """

Delta_c = np.linspace(-100, 500, 1000) # Coupling laser \Delta_c/2\pi [MHz]
alpha = 1.65 # Optical density [1]
Gamma_e = 2 * np.pi * 6 # Spontaneous emission excited state [MHz]
gamma = 2 * np.pi * 5.4 # Coherence dephasing rate between |g> and |r>
Delta_p = 0 # Detuning probe --> we take resonant case


Transmission_Ic = []
Laser_power = []
FWHM_Ic = []
Peaks = []
for Ic in np.linspace(1, 20, 10): # all population in 1 zeeman state
    Omega = np.sqrt(Ic) # The Rabi frequency of the probe laser [MHz]

    #######################################################################

    # 3/2 state - |5>
    CG_5 = 0 #0.5 * 0.632 * 1/3 # Hyper fine, fine CGC and Wigner Eckart element respectively
    Omega_c = CG_5 * 2 * np.pi * Omega # The Rabi frequency of the coupling laser [MHz]
    Delta_s = 0 # EIT transition |33D_3/2>
    delta = Delta_s + Delta_p + Delta_c * (2*np.pi) # Two-photons detuning


    # Absorption cross-section
    sigma_eg_5 = (Gamma_e ** 2 * (4 * gamma ** 2 + 4 * delta ** 2) + 2 * gamma * Omega_c ** 2 * Gamma_e) / ((2 * gamma * Gamma_e + Omega_c ** 2 - 4 * Delta_p * delta) ** 2 + (4 * Delta_p * gamma + 2 * Gamma_e * delta) ** 2)

    #######################################################################

    # 5/2 state - |4>
    CG_4 = 0 #0.189 * 0.316 * 1/3
    Omega_c = CG_4 * 2 * np.pi * Omega # The Rabi frequency of the coupling laser [MHz]
    Delta_s = -2 * np.pi * 336.4 # EIT transition |33D_5/2> --> needs minus sign otherwise symetric at -336.4 MHz
    delta = Delta_s + Delta_p + Delta_c * (2*np.pi) # Two-photons detuning

    # Absorption cross-section
    sigma_eg_4 = (Gamma_e ** 2 * (4 * gamma ** 2 + 4 * delta ** 2) + 2 * gamma * Omega_c ** 2 * Gamma_e) / ((2 * gamma * Gamma_e + Omega_c ** 2 - 4 * Delta_p * delta) ** 2 + (4 * Delta_p * gamma + 2 * Gamma_e * delta) ** 2)

    #######################################################################

    # 5/2 state - |3>
    CG_3 = 1 * 1 * 1/3 # Hyper fine, fine and WE coef.
    Omega_c = CG_3 * 2 * np.pi * Omega # The Rabi frequency of the coupling laser [MHz]
    Delta_s = -2 * np.pi * 336.4 # EIT transition |33D_5/2> --> needs minus sign otherwise symetric at -336.4 MHz
    delta = Delta_s + Delta_p + Delta_c * (2*np.pi) # Two-photons detuning
    print('Previous Omega_c value:', 2 * np.pi * 7.8, 'Our variation:', 2 * np.pi * Omega)
    Laser_power.append(27*((2 * np.pi * Omega) / (2 * np.pi * 7.8))) # Reference coupling power 27mW
    print(Laser_power)


    # Absorption cross-section
    sigma_eg_3 = (Gamma_e ** 2 * (4 * gamma ** 2 + 4 * delta ** 2) + 2 * gamma * Omega_c ** 2 * Gamma_e) / ((2 * gamma * Gamma_e + Omega_c ** 2 - 4 * Delta_p * delta) ** 2 + (4 * Delta_p * gamma + 2 * Gamma_e * delta) ** 2)


    # Transmission Rydberg
    T_3 = np.exp(-alpha*sigma_eg_3)
    T_4 = np.exp(-alpha*sigma_eg_4)
    T_5 = np.exp(-alpha*sigma_eg_5)
    # Transition from |6>
    T_8 = np.exp(-alpha*sigma_eg_3)
    T_9 = np.exp(-alpha*sigma_eg_4)
    T_10 = np.exp(-alpha*sigma_eg_5)
    Tc = (T_5 + T_4 + T_3)/2 + (T_8 + T_9 + T_10)/2
    Transmission_Ic.append(Tc)

    # FWHM
    peaks, _ = find_peaks(Tp)
    results_half = peak_widths(Tp, peaks, rel_height=0.5)
    FWHM = results_half[0]
    print('linewidth',FWHM)
    Peaks.append((max(Tc) - min(Tc)) * 100)  # -min(Tp))
    FWHM_Ic.append(FWHM)

plt.figure(figsize=(8, 4))
plt.text(150, 60.5, r'$|33D_{5/2},mj=\pm5/2, mf=3>$', fontsize=8)
plt.plot(Delta_c, Transmission_Ic[0]*100)
plt.plot(Delta_c, Transmission_Ic[1]*100)
plt.plot(Delta_c, Transmission_Ic[2]*100)
plt.plot(Delta_c, Transmission_Ic[3]*100)
plt.plot(Delta_c, T*100, '--', color='tab:blue')
plt.xlabel(r'Coupling laser power [mW]')
plt.ylabel(r'Transmission [%]')
plt.xlim([Delta_c[0], Delta_c[len(Delta_c)-1]])
plt.legend([str(format(Laser_power[0], '.2f')) + 'mW', str(format(Laser_power[1], '.2f')) + 'mW', str(format(Laser_power[2], '.2f')) + 'mW', str(format(Laser_power[3], '.2f')) + 'mW', '27 mW'])
plt.title(r'Transmission spectrum for different coupling laser power')
plt.ylim([57.5, 61])

plt.figure()
plt.scatter(Laser_power, Peaks)
plt.xlabel('Coupling laser power [mW]')
plt.ylabel('EIT peak intensity [%]')
plt.title('EIT peak intensity vs coupling laser power')
plt.show()


