# Quantum_systems
Repository with the Rydberg atom EIT and N-V center ODMR simulations

The scripts present in this repository are meant to reproduce results presented in the scientific article:
Su et al. ‘Optimizing the Rydberg EIT spectrum in a thermal vapor’ Optics Express 3.2/17 (2022)

The codes present:
EIT_polarization.py --> simulates the EIT peaks amplitude behavior as a function of the polarization of the coupling field
EIT_magnetic_field.py --> simulates the EIT peaks amplitude and splitting as a function of an applied magnetic field
EIT_temperature.py --> simulates the EIT peaks amplitude with respect to temperature and probe field strenght 
Functions.py --> contains the solvers for the OBE
Study_peak_optimization.py --> simulates EIT peaks for two different polarization of the coupling field + look at the influence of the probe detuning on the EIT peak (model limitation)

Note that you can modify the following parameters in the simulation:
alpha : Optical density Rydberg vapor (depends on the temperature system)
Gamma_e : Spontaneous emission excited state [MHz]
Gamma_r : Spontaneous emission Rydberg state [MHz]
Omega_c_CGC1 : The Rabi frequency of the coupling laser for CGC = 1 [MHz]
gamma : Decoherence (interaction Rydberg atom with environment --> atomic noise)

Note: before modifing parameters in the simulation, take a closser look at the reference paper such that the parameters are still within realistic values.



