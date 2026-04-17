# Quantum_systems
Repository with NV center scripts

The scripts present in this repository are meant to simulate fluorescence emission in NV centers
under green laser pump and MW magnetic field.

The codes present in this repository:



1- Energy_states.py: 

It simulates the green laser light applied to NV centers in a diamond sample. In this code, you have the choice between
2 different modalities: confocal microscopy (CM) and evanescent field from WG (EV). The CM modality assumes a bulk diamond
sample for which you can control the size and the NV center density in the section: ### Parameters ### - You can modify
The EV modality assumes nano diamonds deposited on the surface of a WG, you can control the number of nano diamonds,
the concentration of NV centers, the dimensions of the WG and the material of the waveguide through the effective refractive index,
the parameters can be modified in section: ### Parameters ### - You can modify.

An important parameter which should be determined is the intensity of the green laser which can be found in the section: ### Parameters ### - You can modify.
This value will be saved with the data obtained from the Energy_state.py simulation and is directly uploaded in the codes starting
with ODMR_spectrum.

What the code Energy_state.py does is that it creates a system with a certain NV center distribution (depending on parameters
and modality chosen) and calculate the electric field felt by each NV center which will depend on the EV field distribution throughout the 
system (for more information about the physics encoded in the simulation, refer to the pptx N-V_center section "Script, simulation code implementation" 
as well as section "Green laser power estimation"). The information about the green field felt by each NV center is saved and
loaded in the scripts starting with ODMR_spectrum in order to obtain the fluorescence response of each NV center proportional to 
the electric field felt by the NV center. 

In addition to all the above, the script Energy_state.py also calculates the indirect influence of the green laser field
on the local dielectric environment of the diamond sample (aka ions in the lattice) which will create a stark field (proportional
to the green laser field) which influences (very small influence but becomes relevant in weak green laser field regime)
the quantum states of the NV center. This information is also automatically saved and loaded in the scripts ODMR_spectrum
in order to simulate a more accurate fluorescence response. More information about the physics encoded in this simulation
can be found in the pptx N-V_center section "Script, simulation code implementation" in slides with title "Spin Hamiltonian model".

Note: the script Energy_state.py should be run first if you want to do an ODMR simulation --> choice of system and 
laser field intensity.


2- Spectrum_effectiveNV.py: simulates fluorescence spectrum as shown in the reference paper cited in the sript.
Notes:  
-In order to run this code, you do not need to run the script Energy_state.py prior as the model used in 
Spectrum_effectiveNV.py stands on its own with its own parameters.
-Amongst all the scripts I have in this N-V center repository, this script is the simplest implementation of an ODMR spectrum 
as it follows the model used in the reference paper and therefore assumes an effective NV response independent from
any CM or EV modality --> therefore, this code is good for getting a feeling of the behavior of ODMR spectrum with different
parameters however, it does not reflect the real sytem we want to deal with in experiments or on chip.


3- ODMR_spectrum_5levels.py: this script simulates the fluorescence spectrum using the model described in the reference paper cited 
in the script. This scripts needs the data from Energy_states.py to run. Note that this script uses the first NV center
model studied and was adapted to fit the modalities CM and EV. However, keep in mind that this model and therefore the 
results coming from this simulation have limitations. Indeed, I have used a model which gives an ODMR spectrum for an 
effective group of NV centers and I used it for each of my NV center with different 'felt' electric field. This was for me 
to get a feeling of the behavior of each ODMR spectra depending on the NV center I was looking at and also having a look
at the total fluorescence response of the system. Thus, while this simulation gives a feeling of the amplitude of the 
fluorescence response per NV center vs for all NV centers, it has limitations when it comes to studying the fluoresence
response as a function of e.g. the green laser field. For this study, I would use the script 'ODMR_spectrum_3levels.py'
which is a 'home-made' model which is more 'physical' for the system I chose to consider.


4- ODMR_spectrum_3levels.py: basically the same code as ODMR_spectrum_5levels.py but for a description of
the NV center with a 3 levels quantum system.


5- Functions.py: this scripts is as its name defines it, it collects all the functions needed for running all the scripts.
Each


6- ODMR_plot_....py: script which I used to plot my figures. One of them calculate quatities like the
ODMR contrast and linewidth.
