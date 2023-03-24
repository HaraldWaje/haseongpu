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

# VTK Wedge (=13) Data writing
# Daniel Albach                                         2009/05/29

import numpy as np

def vtk_wedge(file_n, phi_ASE, p, t_int, mesh_z, z_mesh):
    
    # open a file
    filename = file_n

    with open(filename, 'wt') as fid: 

        # newline character
        nl = '\n'

        size_p = phi_ASE.shape[0]
        size_p2 = phi_ASE.shape[0] * mesh_z

        fid.write('# vtk DataFile Version 2.0' + nl + 'Wedge example ' + nl + 'ASCII' + nl +
                'DATASET UNSTRUCTURED_GRID' + nl +
                'POINTS ' + str(size_p2) + ' float' + nl)

        # now write the data-coordinates
        # create the cells with the z-coordinates at first, then horizontally concatenate
        h = np.zeros((len(p), 1))
        # first level is at zero level
        h = z_mesh

        for i_z in range(mesh_z):
            v = np.hstack((p, (h * (i_z - 1))))
            for i_v in range(size_p):
                fid.write('{0:f} {1:f} {2:f}'.format(v[i_v, 0], v[i_v, 1], v[i_v, 2]) + nl)

        # now write the cell connections
        fid.write('CELLS {} {}'.format((mesh_z-1)*t_int.shape[0], (mesh_z-1)*7*t_int.shape[0]) + nl)
        
        # in each line: Number of Points points points in c-style indexing
        t_0 = np.zeros((t_int.shape[0], 1), dtype=np.int32) + 6
        t_1 = t_int
        t_2 = np.zeros_like(t_int, dtype=np.int32)
        
        for i_z in range(1, mesh_z):
            t_2 = t_1 + len(p) 
            tp = np.hstack([t_0, t_1, t_2])
            
            for i_v in range(t_int.shape[0]):
                fid.write('{} {} {} {} {} {} {}{}'.format(tp[i_v,0], tp[i_v,1], tp[i_v,2], tp[i_v,3], tp[i_v,4], tp[i_v,5], tp[i_v,6], nl))
            
            t_1 = t_2
        
        # write the cell types
        fid.write('CELL_TYPES {}{}'.format((mesh_z-1)*t_int.shape[0], nl))
        for i_ct in range(t_int.shape[0]*(mesh_z-1)):
            fid.write('13{}'.format(nl))
        
        # write the point data
        fid.write('POINT_DATA {}{}'.format(mesh_z*p.shape[0], nl))
        fid.write('SCALARS scalars float 1{}'.format(nl))
        fid.write('LOOKUP_TABLE default{}'.format(nl))
        
        # write the PHI_ASE data
        for i_z in range(mesh_z):
            v = phi_ASE[:, i_z]
            for i_v in range(size_p):
                fid.write('{:.8f}{}'.format(v[i_v], nl))
    
