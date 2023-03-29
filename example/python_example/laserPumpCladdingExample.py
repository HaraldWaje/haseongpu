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

############################### Laser pump routine ################################
# ASE calulations main file 16kW
# longer calculation without pump action possibel for heat generation
# estimation in the cladding
#
# @author Daniel Albach
# @date   2011-05-03
# @licence: GPLv3
#
###################################################################################
import time
import numpy as np
from scipy.io import loadmat, savemat
from scipy.interpolate import griddata
from beta_int3 import beta_int3
from set_variables import set_variables
from vtk_wedge import vtk_wedge
from calcPhiASE import calcPhiASE

tic= time.perf_counter()

# Crystal parameter
crystal = {
    'doping': 2,
    'length': 0.7,  # [cm]
    'tfluo': 9.5e-4,  # 1.0e-3 pour DA
    'nlexp': 1,
    'levels': 10,
}
# nonlinearity of the doping - exponential - factor front/end of the
# crystal

# Timesteps
steps = {
    'time': 100,
    'crys': crystal['levels'],
}

# Pump parameter
pump_stretch = 1
pump_aspect = 1
pump_diameter = 3
pump = {
    's_abs': 0.778e-20,  # Absorption cross section in cm^2 (0.778e-20 pour DA)
    's_ems': 0.195e-20,  # Emission cross section in cm^2 (0.195e-20 pour DA)
    'I': 16e3,  # Pump intensity in W/cm^2
    'T': 1e-3,  # Pump duration in s
    'wavelength': 940e-9,  # Pump wavelength in m
    'ry': pump_diameter / 2 * pump_aspect,
    'rx': pump_diameter / 2,  # Pump radius
    'exp': 40,  # Pump exponent
}
# Laser parameter
laser = {
    's_abs' : np.loadtxt('sigma_a.txt'),  # Absorption spectrum cm2(1.16e-21 pour DA)
    's_ems' : np.loadtxt('sigma_e.txt'), # Emission spectrum in cm2(2.48e-20)
    'l_abs' : np.loadtxt('lambda_a.txt'), # Wavelengths absoroption spectrum in nm (x values for absorption)
    'l_ems' : np.loadtxt('lambda_e.txt'), # Wavelengths emission spectrum in nm (y values for emission)
    'l_res' : 1000,                      # Resolution of linear interpolated spectrum
    'I' : 1e6,                           # Laser intensity
    'T' : 1e-8,                          # Laser duration
    'wavelength' : 1030e-9,              # Lasing wavelength in m
}
laser['max_ems'] = np.max(laser['s_ems'])
laser['i'] = np.argmax(laser['s_ems'])
laser['max_abs']= laser['s_abs'][laser['i']]

# Mode parameter
mode = {
    'BRM': 1,  # 1 Backreflection, 0 Transmissionmode
    'R': 1,    # reflectivity of the coating
    'extr': 0,  # no extraction!!
}

# Constants
phy_const= {
    'N1per': 1.38e20,
    'c': 3e8,
    'h': 6.626e-34,
}

N_tot = phy_const['N1per'] * crystal['doping']
Ntot_gradient = np.zeros(crystal['levels'])   # doping gradient if not given by load!
Ntot_gradient[:] = crystal['doping'] * phy_const['N1per']
mesh_z = crystal['levels']
z_mesh = crystal['length'] / (mesh_z-1)
timeslice = 50
timeslice_tot = 150
timetotal = 1e-3  # [s]
time_t = timetotal / timeslice

# ASE application
maxGPUs = 1 #TODO this was 2 before but gave error
nPerNode = 4
deviceMode = 'gpu'
# parallelMode = 'mpi'
parallelMode = 'threaded'
useReflections = True
refractiveIndices = [1.83, 1, 1.83, 1]
repetitions = 4
minRaysPerSample = 1e5
maxRaysPerSample = minRaysPerSample * 100
mseThreshold = 0.005


# Constants for short use
c = phy_const['c']  # m/s
h = phy_const['h']  # Js

################################ Create mesh ######################################%
# Load the grid from file
# the grid is done using the distmesh routines
# use e.g. cr_60mm_30mm.m in meshing folder

# load data from .mat files
pt_data = loadmat('pt.mat')

# extract necessary variables
p = np.array(pt_data['p'])
t = np.array([[x -1 for x in y] for y in  pt_data['t']]) # indes shift due to earlier matlab konvention 
set_variables(p, t)

variable_data = loadmat('variable.mat')

normals_x= variable_data['normals_x']
normals_y= variable_data['normals_y']
normals_z= variable_data['normals_z']
ordered_int= variable_data['ordered_int']
t_int= variable_data['t_int']
x_center= variable_data['x_center'][0]
y_center= variable_data['y_center'][0]
surface= variable_data['surface']
normals_p= variable_data['normals_p']
forbidden=variable_data['forbidden']

beta_cell = np.zeros((p.shape[0], mesh_z))
beta_vol = np.zeros((t.shape[0], mesh_z-1))
nT, b = ordered_int.shape                
reflectivities = np.zeros((1, nT*2))

# initialize arrays
phi_ASE = np.zeros((p.shape[0], mesh_z))
dndt_ASE = np.copy(phi_ASE)
flux_clad = np.copy(phi_ASE)
dndt_pump = np.copy(beta_cell)

# save initial values to file
vtk_wedge('beta_cell_0.vtk', beta_cell, p, t_int, mesh_z, z_mesh)
vtk_wedge('dndt_pump_0.vtk', dndt_pump, p, t_int, mesh_z, z_mesh)
vtk_wedge('dndt_ASE_0.vtk', dndt_ASE, p, t_int, mesh_z, z_mesh)
savemat('save_0.mat', {'pys_const': phy_const, 'mode': mode, 'steps': steps, 'p': p })

# simulate pumping and ASE for each point in the mesh
temp = pump['T']
time_beta = 1e-6
pump['T'] = time_beta
temp_f = crystal['tfluo']
crystal['tfluo'] = 1
beta_c_2 = np.zeros((p.shape[0], mesh_z))
intensity = pump['I']

for i_p in range(p.shape[0]):
    beta_crystal = beta_cell[i_p, :]
    pulse = np.zeros((steps['time'], 1))
    pump['I'] = intensity * np.exp(-np.sqrt(p[i_p, 0]**2/pump['ry']**2 + p[i_p, 1]**2/pump['rx']**2)**pump['exp'])
    beta_crystal, beta_store, pulse, Ntot_gradient = beta_int3(beta_crystal, pulse, phy_const, crystal, steps, pump, mode, Ntot_gradient)
    beta_c_2[i_p, :] = beta_crystal[:]

# calculate change in beta_cell due to pumping
dndt_pump = (beta_c_2 - beta_cell) / time_beta

# reset parameters to initial values
pump['I'] = intensity
pump['T'] = temp
crystal['tfluo'] = temp_f
time_int = 1e-16

for i_p in range(p.shape[0]): 
    for i_z in range(mesh_z):
        beta_cell[i_p, i_z] = crystal['tfluo'] * (dndt_pump[i_p,i_z] - dndt_ASE[i_p,i_z]) * (1-np.exp(-time_t/crystal['tfluo'])) + beta_cell[i_p,i_z]*np.exp(-time_t/crystal['tfluo'])

vtk_wedge('beta_cell_1.vtk', beta_cell, p, t_int, mesh_z, z_mesh) 

################################ Definition of cladding ###########################
# Which index is the cladding?
clad_number = 1
clad_number = np.int32(clad_number)
numberOfTriangles, b = t_int.shape
clad = np.zeros((numberOfTriangles,1))

# Absorption of the cladding
clad_abs = 5.5

# Prepare which part is concerning the cladding and which one is the Yb doped part
# this part mus be adapted for each case!
clad_index = np.where(clad==clad_number)[0]
Yb_index = np.where(clad!=clad_number)[0]

# the logic is inverse, which index to delete ;)
# this represents the triangle index and using the p-variable, you get the points...
Yb_points_t = t.copy()
Yb_points_t = np.delete(Yb_points_t, clad_index, axis=0)
Yb_points_t2 = np.reshape(Yb_points_t,(-1,1))
del Yb_points_t
Yb_points_t = np.unique(Yb_points_t2)
del Yb_points_t2

clad_points_t = t.copy()
clad_points_t = np.delete(clad_points_t, Yb_index, axis=0)
clad_points_t2 = np.reshape(clad_points_t,(-1,1))
del clad_points_t
clad_points_t = np.unique(clad_points_t2)
del clad_points_t2

#Now get the lengths of the
length_clad = clad_points_t.shape[0]
length_Yb = Yb_points_t.shape[0]

#Make sure that clad is integer variable
clad_int = np.int32(clad)

################################ Main pump loop ###################################

for i_slice in range(1, timeslice_tot):
    print('TimeSlice', i_slice, 'of', timeslice_tot-1, 'started')
    # ******************* BETA PUMP TEST ******************************
    # make a test with the gain routine "gain.m" for each of the points
    # define the intensity at the nominal points of the grid and make the
    # pumping to get the temporal evolution of the beta
    # make a rectangular field, make the interpolation of the points on it
    # if we don't have any formula discribing the intensity distribution
    # as first test do a supergaussian distribution for 1 s and then estimate
    # the dndt|pump

    # Write output for this timeslice to file
    file_b = 'beta_cell_' + str(i_slice) + '.vtk'
    file_p = 'dndt_pump_' + str(i_slice) + '.vtk'
    file_A = 'dndt_ASE_' + str(i_slice) + '.vtk'
    file_C = 'flux_clad_' + str(i_slice) + '.vtk'

    vtk_wedge(file_b, beta_cell, p, t_int, mesh_z, z_mesh)
    vtk_wedge(file_p, dndt_pump, p, t_int, mesh_z, z_mesh)

    # Interpolate beta_vol from beta_cell
    x_1 = p[:,0]
    y_1 = p[:,1]
    x_2 = p[:,0]
    y_2 = p[:,1]

    x = np.concatenate((x_1,x_2))
    y = np.concatenate((y_1,y_2))

    z_1 = np.zeros(x_1.shape[0])
    z_2 = np.zeros(x_2.shape[0])
    z_2 = z_2 + z_mesh

    z = np.concatenate((z_1,z_2))

    xi = x_center
    yi = y_center
    zi = np.zeros(xi.shape)
    zi = zi + z_mesh/2

    for i_z in range(mesh_z-1):
        v_1 = beta_cell[:,i_z]
        v_2 = beta_cell[:,i_z+1]

        v = np.concatenate((v_1,v_2))
        pts= np.array([x, y, z])
        ges= np.array([xi,yi,zi]) 

        beta_vol[:,i_z] = griddata((x,y,z), v, (xi,yi,zi))
        #beta_vol[:,i_z] = griddata(pts, v, ges)

        z = z + z_mesh
        zi = zi + z_mesh

    ############################ Call external ASE application ####################
    calcPhiASE(
        p,
        t_int,
        beta_cell,
        beta_vol,
        clad_int,
        clad_number,
        clad_abs,
        useReflections,
        refractiveIndices,
        reflectivities,
        normals_x,
        normals_y,
        ordered_int,
        surface,
        x_center,
        y_center,
        normals_p,
        forbidden,
        minRaysPerSample,
        maxRaysPerSample,
        mseThreshold,
        repetitions,
        N_tot,
        z_mesh,
        laser,
        crystal,
        mesh_z,
        deviceMode,
        parallelMode,
        maxGPUs,
        nPerNode)
    
        # Calc dn/dt ASE, change it to only the Yb:YAG points
        # For Yb points
    for i_p in range(length_Yb):
        for i_z in range(mesh_z):
            pos_Yb = Yb_points_t[i_p]
            # Calc local gain (g_l)
            g_l = -(N_tot*laser['max_abs'] - N_tot*beta_cell[pos_Yb,i_z]*(laser['max_ems']+laser['max_abs']))
            dndt_ASE[pos_Yb,i_z] = g_l * phi_ASE[pos_Yb,i_z] / crystal['tfluo']

    # Now for the cladding points
    for i_p in range(length_clad):
        for i_z in range(mesh_z):
            # Calc local gain (g_l)
            pos_clad = clad_points_t[i_p]
            flux_clad[pos_clad,i_z] = clad_abs*phi_ASE[pos_clad,i_z]/crystal['tfluo']

    vtk_wedge(file_A,dndt_ASE , p, t_int, mesh_z, z_mesh)
    vtk_wedge(file_C,flux_clad , p, t_int, mesh_z, z_mesh)
    np.savez_compressed('save_' + str(i_slice) + '.npz', dndt_ASE=dndt_ASE, flux_clad=flux_clad)

    ############################ Prepare next timeslice ###########################
    temp = pump['T']
    time_beta = 1e-6
    pump['T'] = time_beta
    temp_f = crystal['tfluo']
    crystal['tfluo'] = 1
    beta_c_2 = np.zeros((p.shape[0],mesh_z))
    intensity = pump['I']

    for i_p in range(p.shape[0]):
        for i_z in range(mesh_z): 
            beta_cell[i_p,i_z] = crystal['tfluo']*(dndt_pump[i_p,i_z]-dndt_ASE[i_p,i_z])*(1-np.exp(-time_t/crystal['tfluo']))+beta_cell[i_p,i_z]*np.exp(-time_t/crystal['tfluo'])

    file_b = ['beta_cell_' + str(timeslice_tot) + '.vtk']
    file_p = ['dndt_pump_' + str(timeslice_tot) + '.vtk']
    file_A = ['dndt_ASE_' + str(timeslice_tot) + '.vtk']
    file_C = ['flux_clad_' + str(timeslice_tot) + '.vtk']

    vtk_wedge(file_b, beta_cell, p, t_int, mesh_z, z_mesh)
    vtk_wedge(file_p, dndt_pump, p, t_int, mesh_z, z_mesh)
    vtk_wedge(file_A, dndt_ASE, p, t_int, mesh_z, z_mesh)
    vtk_wedge(file_C, flux_clad, p, t_int, mesh_z, z_mesh)

    print('Calculations finished')
    toc = time.perf_counter()
    elapsed_time= toc - tic 

    #print(f'Time taken: {toc} seconds')

    # Obviously I forgot to save the latest results, must be added!