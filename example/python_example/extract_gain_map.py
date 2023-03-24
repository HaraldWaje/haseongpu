
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
 # but WITHOUT ANY WARRANTY; without even the implied warranty of
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 # GNU General Public License for more details.
 #
 # You should have received a copy of the GNU General Public License
 # along with HASEonGPU.
 # If not, see <http://www.gnu.org/licenses/>.
 #################################################################################

# extraction of gain map out of the calculation
# just make it in the center - position

# at first try with one distribution

import numpy as np
import scipy
from scipy.interpolate import griddata
import time
import matplotlib.pyplot as plt
import beta_int

tic = time.perf_counter()

timeslices_tot = 150
grid_min= -3
grid_max= 3.5
grid_step= 0.5
central_slot= 7
x_grid,y_grid= np.meshgrid(np.arange(grid_min, grid_max, grid_step), np.arange(grid_min, grid_max, grid_step))
gainmap_Interp= np.zeros(len(x_grid), len(y_grid), timeslices_tot)
gain_line= np.zeros(timeslices_tot)

#load var with const values 
file_const_val= "save_0.mat"
const_data = scipy.io.loadmat(file_const_val)
steps= const_data['steps']
mode= const_data['mode']
phy_const= const_data['phy_const']
p= const_data['p']

for i_s in range(timeslices_tot):
    print(f"TimeSlice {i_s}")
    plt.draw()
    
    i_ss = i_s - 1

    #load var with shifting values
    filename = f"save_{i_ss}.mat"
    data = scipy.io.loadmat(filename)

    beta_cell= data['beta_cell']
    crystal= const_data['crystal']
    laser= const_data['laser']
    pump= const_data['pump']

    time_step = pump["T"] / (steps["time"] - 1)
    crystal_step = crystal["length"] / (steps["crys"] - 1)
    
    gainmap = np.zeros((len(p), 1))

    for i_p in range(len(p)):

        beta_crystal[1,:] = beta_cell[i_p,:]

        laser['I'] = 1

        # grids for plotting the lines
        grid_t = np.arange(0, pump['T'] + time_step, time_step)

        grid_z = np.arrange(0, crystal['length']+ crystal_step, crystal_step)

        time_step = pump['T']/(steps['time']-1)
        crystal_step = crystal['length'] / (steps['crys']-1) 

        gain_local = np.zeros(steps['crys'],1)

        pulse = np.zeros(steps['time'],1)
        pulse = laser['I']

        # extraction mode #TODO Check if this is necessary
        mode['BRM'] = 1 # 1 Backreflection, 0 Transmissionmode
        mode['R'] = 1 # reflectivity of the coating
        mode['extr'] =1

        # laser wavelength
        laser['max_ems'], i = np.max(laser['s_ems']), np.maxarg(laser['s_ems'])
        laser['max_abs'] = laser['s_abs'][i]

        sigma_abs_L=laser['max_abs']
        sigma_ems_L=laser['max_ems']
        beta_min = np.zeros(25,1)
        beta_min = sigma_abs_L/(sigma_abs_L+sigma_ems_L)
        grid_z_beta_min = np.arrange(0, crystal['length'] * 25/ 24, crystal['length'])

        # time constants for the pump
        time_step_ex = laser['T'] / (steps['time']-1)
        grid_t_ex = np.arrange(0, laser['T'] + time_step_ex, time_step_ex)

        energy_pulse = scipy.integrate.trapz(pulse, x= grid_t_ex)

        # energy dump for the plot
        #energy_pulse_round[1,1]=energy_pulse
        energy_pulse_round=energy_pulse   #TODO is array necessary? 

        # integration yields the energy density, this has to be multiplied with the
        # "average" surface of the pumped area
        # energy_density.before = trapz(grid_z,beta_crystal.*Ntot_gradient)*(h*c/pump.wavelength);
        # energy_density_ex.before = trapz(grid_z,(beta_crystal-beta_min(1)).*Ntot_gradient)*(h*c/pump.wavelength);
    #     beta_crystal_l = beta_crystal;
    #     beta_store_l=beta_store;

    #     for iroundtrips=1:1
        # this is the function call to the extraction
        # we have to call it two times seperately for the aller-retour!
        [beta_crystal,beta_store,pulse] = beta_int(beta_crystal,pulse,phy_const,crystal,steps,laser,mode)
    #     gain_local(:) = -(sigma_abs_L - beta_crystal(:)*(sigma_abs_L+sigma_ems_L)).*Ntot_gradient(:);

        gainmap[i_p,1] = pulse[1]/laser['I']

        # now the pulse is at the backside

        # energy_pulse = trapz(grid_t_ex,pulse);
        # energy_pulse_round(iroundtrips+1,1)=energy_pulse;
    #     pulse_round(:,iroundtrips)=pulse(:);
        # after each pass he sees some losses e.g. 5#
        # now make the losses, but be carefull - the position changes the energy
        # output!!!
        # pulse(:)=pulse(:).*mode.R_ex;
    #     end
        # integration yields the energy density, this has to be multiplied with the
        # "average" surface of the pumped area
        # energy_density.after = trapz(grid_z,beta_crystal_l.*Ntot_gradient)*(h*c/pump.wavelength);

        # bar (energy_pulse_round); figure(gcf)

        
    #     figure(2)

    gainmap_Interp[:, :, i_s] = griddata(p[:,0], p[:,1], gainmap[:,0], x_grid, y_grid)

        
    #     imagesc(gainmap_Interp);
    #     axis equal;
    #     colorbar;

# now make an interpolation to make it plotable

# extract data
for i_t in range (timeslices_tot):
    gain_line[i_t]=gainmap_Interp(central_slot, central_slot,i_t) #TODO get rid of 7

with open('gain_line.txt','w') as x: 
    for item in gain_line:
        x.write('%.50f\n' % item)
        
toc= time.perf_counter()
elapsed_time= toc - tic 
