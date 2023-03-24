#################################################################################
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

# function set_variables
# compressed version of mesh_cyl_rect which creates and extracts the
# informations needed for the further work on the data
# the variables created are:
# 'normals_x','normals_y','normals_z','ordered_int','x_center','y_center','surface','normals_p','forbidden'
# in the corresponding order:
# normals_x,y,z: the normals on the faces, thereby we implied, that we have a rectilinear meshing in z
# ordered : the neighboring triangles index , if you want to get from one to another by raytracing
# x,y_center: the central positions of the cells
# surface : values of the surfaces - important later on for the importance sampling
# normals_p : the index to one of the points used to calculate the normals of the traingle faces
# forbidden : the index of the crossed face for the next  triangle to enter, that a checking of this surface is ommited

# at the end a file called "variables.mat" with all the variables is
# written, all file convervions are done and c-style indexing is applyed
# for the indexing files (matlab indexing - 1 = c-indexing)

import numpy as np
from scipy.io import savemat

def set_variables(p, t):
    # calculate the index of the neighbor cells
    length_t = len(t)
    ordered = np.zeros((length_t, 3))

    # now search the cells which have two of the three points in common
    for i_point in range(length_t):
        point_1 = t[i_point, 0]
        point_2 = t[i_point, 1]
        point_3 = t[i_point, 2]
        # search for the combination of point 1 and 2 out of the original
        for i_test in range(length_t):
            if (t[i_test, 0] == point_1) or (t[i_test, 1] == point_1) or (t[i_test, 2] == point_1):
                if (t[i_test, 0] == point_2) or (t[i_test, 1] == point_2) or (t[i_test, 2] == point_2):
                    if i_test != i_point:
                        ordered[i_point, 0] = i_test
                if (t[i_test, 0] == point_3) or (t[i_test, 1] == point_3) or (t[i_test, 2] == point_3):
                    if i_test != i_point:
                        ordered[i_point, 1] = i_test
            if (t[i_test, 0] == point_3) or (t[i_test, 1] == point_3) or (t[i_test, 2] == point_3):
                if (t(i_test,1)==point_2) or (t(i_test,2)==point_2) or (t(i_test,3)==point_2):
                    if (i_test != i_point):
                        ordered[i_point,2] = i_test

    # make a list with the forbidden faces of the triangles
    # these are faces, which are connecting faces between the triangles,
    # meaning: face a of triangle n is face c of triangle l
    # remember the definition: face 1: p1-2, face 2: p1-3, face 3: p2-3
    forbidden = np.zeros((length_t, 3))  #TODO maybe use elif in loops to minimize time? 
    for i_point in range(length_t):     
        face_1 = ordered[i_point, 0]
        if (face_1 == 0):
            forbidden[i_point, 0] = 0
            continue
        
        if (ordered[face_1, 1] == i_point):
            forbidden[i_point, 0] = 1

        if (ordered[face_1, 2] == i_point):
            forbidden[i_point, 0] = 2

        if (ordered[face_1, 3] == i_point):
            forbidden[i_point, 0] = 3

    for i_point in range(length_t):
        face_2 = ordered[i_point, 2]
        if (face_2==0):
            forbidden[i_point, 1] = 0
            continue

        if (ordered[face_2, 1]==i_point):
            forbidden[i_point-1, 1] = 1

        if (ordered[face_2, 2]==i_point):
            forbidden[i_point-1, 1] = 2

        if (ordered[face_2, 1]==i_point):
            forbidden[i_point-1, 1] = 3
            
    for i_point in range(length_t):
        face_3 = ordered[i_point, 3]
        if (face_3 == 0):
            forbidden[i_point, 3] = 0
            continue

        if (ordered[face_3, 1]==i_point):
            forbidden[i_point,3]=1

        if (ordered[face_3, 2]==i_point):
            forbidden[i_point,3]=2

        if (ordered[face_3, 3]==i_point):
            forbidden[i_point,3]=3


        # calculate the centers of the triangles via barycentric coordinates
        center_vec = [1/3, 1/3, 1/3]

        x_center = np.zeros(length_t,1)
        y_center = np.zeros(length_t,1)

    for i_points in range(length_t):

        Px = [p[t[i_points,0], 0], p[t[i_points,1], 0], p[t[i_points,2], 0]]
        Py = [p[t[i_points,0], 1], p[t[i_points,1], 1], p[t[i_points,2], 1]]
        
        x_center[i_points,0]=Px*center_vec
        y_center[i_points,1]=Py*center_vec 

        x_center[i_points,0] = np.dot(Px, center_vec)
        y_center[i_points,0] = np.dot(Py, center_vec)
    # # now plot it
    # hold on
    # trimesh(t,p(:,1),p(:,2),zeros(size(p,1),1))
    # view(2),axis equal,axis off,drawnow
    # plot(x_center,y_center,'*')
    # hold off

        # calculate the normal vectors of the different combinations and give a vector with one of the points
    normals_x = np.zeros(length_t,3)
    normals_y = np.zeros(length_t,3)
    normals_z = np.zeros(length_t,3)

    normals_p = np.zeros(length_t,3)

    for i_points in range (length_t): 
        #     normals to points 1-2

        vec_1 = np.array([p[t[i_points,0], 0], p[t[i_points,0], 1], 0]) - np.array([p[t[i_points,1], 0], p[t[i_points,1], 1], 0])
        vec_2 = np.array([p[t[i_points,0], 0], p[t[i_points,0], 1], 0]) - np.array([p[t[i_points,0], 0], p[t[i_points,0], 1], 0.1])
        vec_cross = np.cross(vec_1, vec_2)
        vec_cross = vec_cross / np.linalg.norm(vec_cross)
        normals_x[i_points,0] = vec_cross[0]
        normals_y[i_points,0] = vec_cross[1]
        normals_z[i_points,0] = vec_cross[2]
        normals_p[i_points,0] = t[i_points,0]
            
        #     normals to points 1-3
        vec_1 = np.array([p[t[i_points, 0], 0], p[t[i_points, 0], 1], 0]) - np.array([p[t[i_points, 2], 0], p[t[i_points, 2], 1], 0])
        vec_2 = np.array([p[t[i_points, 0], 0], p[t[i_points, 0], 1], 0]) - np.array([p[t[i_points, 0], 0], p[t[i_points, 0], 1], 0.1])
        vec_cross = np.cross(vec_1, vec_2)
        vec_cross = vec_cross / np.linalg.norm(vec_cross)
        normals_x[i_points, 1] = vec_cross[0]
        normals_y[i_points, 1] = vec_cross[1]
        normals_z[i_points, 1] = vec_cross[2]
        normals_p[i_points, 1] = t[i_points, 0]


        #     normals to points 2-3
        vec_1 = np.array([p[t[i_points, 1], 0] - p[t[i_points, 2], 0], p[t[i_points, 1], 1] - p[t[i_points, 2], 1], 0])
        vec_2 = np.array([p[t[i_points, 1], 0] - p[t[i_points, 2], 0], p[t[i_points, 1], 1] - p[t[i_points, 2], 1], -0.1])
        vec_cross = np.cross(vec_1, vec_2)
        vec_cross = vec_cross / np.linalg.norm(vec_cross)
        normals_x[i_points, 2] = vec_cross[0]
        normals_y[i_points, 2] = vec_cross[1]
        normals_z[i_points, 2] = vec_cross[2]
        normals_p[i_points, 2] = t[i_points, 1]


    # # 2d interpolation of the new points in the grid
    # z = zeros(size(p,1),1)
    # # as an example make a linear function as the functional values
    # z(:) = 3*p(:,1)
    # 
    # interpolated = griddata(p(:,1),p(:,2),z,x_center,y_center)
    # 
    # # # now show the interpolation error
    # # plot(interpolated-3*x_center(:,1))

    # calculate the surface of the triangles
    surface = np.zeros(len(t),1)

    # Compute the surface area for each triangle
    for i_point in range(len(t)):
        a = ((p[t[i_point, 0], 0] - p[t[i_point, 1], 0]) ** 2 + (p[t[i_point, 0], 1] - p[t[i_point, 1], 1]) ** 2)
        b = ((p[t[i_point, 2], 0] - p[t[i_point, 1], 0]) ** 2 + (p[t[i_point, 2], 1] - p[t[i_point, 1], 1]) ** 2)
        c = ((p[t[i_point, 0], 0] - p[t[i_point, 2], 0]) ** 2 + (p[t[i_point, 0], 1] - p[t[i_point, 2], 1]) ** 2)

        surface[i_point] = np.sqrt(1 / 16 * (4 * a * c - (a + c - b) ** 2))

    # Convert variables to the appropriate data type
    t= np.array(t) 
    t_int = t.astype(np.int32) 
    ordered_int = ordered.astype(np.int32)
    normals_p = normals_p.astype(np.int32)
    forbidden = forbidden.astype(np.int32) #TODO Check if conversion and -1 can be done at once

    # Update the normals_p, t_int, orderedint, and forbidden arrays
    normals_p -= 1
    t_int -= 1
    ordered_int -= 1
    forbidden -= 1

    # Save the variables to a MAT file
    savefile = 'variable.mat'
    savemat(savefile, {'normals_x': normals_x,
                    'normals_y': normals_y,
                    'normals_z': normals_z,
                    'ordered_int': ordered_int,
                    't_int': t_int,
                    'x_center': x_center,
                    'y_center': y_center,
                    'surface': surface,
                    'normals_p': normals_p,
                    'forbidden': forbidden})