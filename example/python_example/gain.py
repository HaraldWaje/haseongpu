 ##################################################################################
 # Copyright 2013 Daniel Albach, Erik Zenker, Carlchristian Eckert
 #
 # This file is part of HASEonGPU
 #
 # HASEonGPU is free software: you can redistribute it and/or modify
 # it under the terms of the GNU General Public License as published by
 # the Free Software Foundation, either version 3 of the License, or
 # (at your option) any later version.
 #
 # HASEonGPU is distributed in the hope that it will be useful,
 # but WITHOUT ANY WARRANTY without even the implied warranty of
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 # GNU General Public License for more details.
 #
 # You should have received a copy of the GNU General Public License
 # along with HASEonGPU.
 # If not, see <http://www.gnu.org/licenses/>.
 #################################################################################

# This will calculate the gain distribution within an Yb:YAG-Crystal
# Monochromatic approach                Daniel Albach
#                   last edit:  2008/10/24
import numpy as np
import time 
import matplotlib.pyplot as plt
import beta_int

tic= time.perf_counter()
#constants
phy_const = {'N1per': 1.38e20, 'c': 3e8, 'h': 6.626e-34}
# crystal
crystal = {'doping': 2, 'length': 0.7, 'tfluo': 9.5e-4, 'nlexp': 1}
# nonlinearity of the doping - exponential - factor front/end of the
# crystal

# steps
steps = {'time': 1000, 'crys': 300}

# In later versions this should be replaced by a vector for the emission
# and absorption cross sections as well for the input spectrum

# pump
pump = {'s_abs': 0.76e-20, #absorption cross section in cm� (0.778e-20 pour DA)
        's_ems': 0.220e-20, #emission cross section in cm�(0.195e-20 pour DA)
        'I': 10e3, #Pump intensity in W/cm�
        'T': 1e-3, #pump duration in s
        'wavelength': 940e-9, #pump wavelength in m
}

# laser
laser = {'s_abs': 1.10e-21, #cm2(1.16e-21 pour DA)
         's_ems': 2.40e-20, #cm2(2.48e-20)
         'I': 1e6, #laser intensity
         'T': 1e-8, #laser duration
         'wavelength': 1030e-9, #lasing wavelength in m
}

# modus definitions
mode = {'BRM': 1, #1 Backreflection, 0 Transmissionmode
        'R': 1, #reflectivity of the coating
}

# constants for short use
c = phy_const['c'] #m/s
h = phy_const['h'] #Js

#  prepare the beta_crystal as a vector to be a parameter, global scope!
beta_crystal = np.zeros((steps['crys'],1))
pulse = np.zeros((steps['time'],1))

time_step = pump.T/(steps['time']-1)
crystal_step = crystal.length/(steps['crys']-1) 

gain_local = np.zeros((steps['crys'],1))

# this is the function call to the pump
beta_crystal,beta_store,pulse,Ntot_gradient = beta_int(beta_crystal,pulse,phy_const,crystal,steps,pump,mode)

# grids for plotting the lines
grid_t = np.arange(0, pump['T'] + time_step, time_step)
grid_z = np.arange(0, crystal['length'] + crystal_step, crystal_step)


# laser wavelength
sigma_abs_L=1.16e-21
sigma_ems_L=2.48e-20
beta_min = np.zeros(25,1)
beta_min = sigma_abs_L/(sigma_abs_L+sigma_ems_L)
grid_z_beta_min = np.arrange(0, crystal['length'] + crystal_step, crystal_step)

# integration yields the energy density, this has to be multiplied with the
# "average" surface of the pumped area
energy_density= {   'before': np.trapz(beta_crystal*Ntot_gradient,grid_z,)*(h*c/pump['wavelength']),
                    'ex_before': np.trapz((beta_crystal-beta_min[0])*Ntot_gradient,grid_z)*(h*c/pump['wavelength']) 
}    

# overwrite the pulse with zeros
pulse = np.zeros((steps['time'],1))

# this is the function call to the extraction
beta_crystal_l,beta_store_l,pulse = beta_int(beta_crystal,pulse,phy_const,crystal,steps,laser,mode)

# integration yields the energy density, this has to be multiplied with the
# "average" surface of the pumped area
energy_density['after']= np.trapz(beta_crystal_l*Ntot_gradient, grid_z)*(h*c/pump.wavelength)

# get the optimum length at the end of the pump time
beta_fit_comp = np.polyfit((grid_z),beta_crystal,4)  # transpose grid_z was used in original matlab code 
beta_fit = np.polyval(beta_fit_comp,grid_z)

intersect=[beta_fit_comp[1], beta_fit_comp[2], beta_fit_comp[3], beta_fit_comp[4], beta_fit_comp[5]-beta_min[1]]
roots_beta = np.roots(intersect)

intersection = 0

for iroots in range(len(roots_beta)): 
    if np.isreal(roots_beta(iroots)) and  roots_beta(iroots) > 0:
        intersection = roots_beta(iroots)

toc= time.perf_counter()
elapsed_time = (toc - tic)
# now calculate the local gain at each point
gain_local = -(sigma_abs_L - beta_crystal*(sigma_abs_L+sigma_ems_L)) * Ntot_gradient

# first image
plt.figure(1)
fig_1, = plt.plot(grid_z, beta_crystal, '-', grid_z_beta_min, beta_min, '--')
plt.title(r'$\beta$ distribution in absense of ASE', fontsize=16)
plt.xlabel('crystal thickness [cm]', fontsize=16)
plt.ylabel(r'$\beta$', fontsize=16)
plt.xlim([0, crystal['length']])
plt.text(0.02, 0.025, r'$\uparrow \beta_{min}$ for 1030nm', fontsize=14)
plt.text(0.2, 0.3, str(crystal['doping']) + '#doping', fontsize=14)
plt.setp(fig_1, linewidth=2)
plt.gca().set_fontsize(13)
plt.gca().set_tick_params(direction='out', length=2, which='both')
plt.gca().set_xmargin(0)
plt.gca().set_ymargin(0)
plt.gca().set_xticks(np.arange(0, crystal['length'] + 1, 0.5))
plt.gca().set_yticks(np.arange(0, 1.1, 0.1))
plt.gca().set_xlim([0, crystal['length']])
plt.gca().set_ylim([0, 1])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# now the beta after the run with the laser
plt.figure(4)
plt.plot(grid_z, beta_crystal_l, '-', grid_z_beta_min, beta_min, '--')
plt.title(r'$\beta$ distribution after one pass')
plt.xlabel('crystal length [cm]')
plt.ylabel(r'$\beta$')
plt.xlim([0, crystal['length']])
plt.text(0.02, 0.025, r'$\uparrow \beta_{min} for 1030nm$')
plt.text(0.2, 0.2, str(crystal['doping']) + '%doping')

fig, haxes = plt.subplots()
hline1, = haxes.plot(grid_z, beta_crystal, linewidth=2)
hline2, = haxes.plot(grid_z, gain_local, linewidth=2)
plt.title(r'$\beta$ distribution in absence of ASE', fontsize=16)
plt.xlabel('crystal thickness [cm]', fontsize=16)
haxes.set_ylabel(r'$\beta$ (z)', fontsize=14)
haxes.set_xlim([0, crystal['length']])
haxes2 = haxes.twinx()
haxes2.set_ylabel('gain (z)', fontsize=14)
haxes2.set_xlim([0, crystal['length']])
hlines = [hline1, hline2]
haxes.tick_params(direction='out', length=2, which='both', axis='both')
haxes.minorticks_on()
haxes2.tick_params(direction='out', length=2, which='both', axis='both')
haxes2.minorticks_on()
plt.setp(haxes, fontsize=13, fontname='Calibri')

# 
# 
# # grid for second plot
# grid_chart_x = 0:crystal_step:crystal.length
# grid_chart_y = pump.T*1000:-time_step*1000:0
# 
# # second image
# figure(3)
# beta_store = rot90(beta_store)
# imagesc(grid_chart_x,grid_chart_y,beta_store)figure(gcf)
# title('Evolution of \beta in absense of ASE as afunction of time and position within the crystal')
# xlabel('position in the crystal [cm]')
# ylabel('time [ms]')
# axis xy
