####################################################################################
# Copyright 2014 Erik Zenker, Carlchristian Eckert
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
###################################################################################

import subprocess 
import numpy as np
import os
import warnings
import shutil 

############################## clean_IO_files #################################
#
# deletes the temporary folder and the dndt_ASE.txt
# 
# @param TMP_FOLDER the folder to remove
#
#
###############################################################################
def clean_IO_files (TMP_FOLDER):

  if os.path.exists(TMP_FOLDER) and os.path.isdir(TMP_FOLDER):
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      shutil.rmtree(TMP_FOLDER)


# ################## parse_calcPhiASE_output #############################
#
# takes all the variables and puts them into textfiles
# so the CUDA code can parse them. Take care that the
# names of the textfiles match those in the parsing function
# 
# for most parameters, see calcPhiASE (above)
# @param FOLDER the folder in which to create the input files (usually a
#               temporary folder visible by all the participating nodes)
#
#
##########################################################################

def create_calcPhiASE_input (points,
                            triangleNormalsX,  
                            triangleNormalsY,   
                            forbiddenEdge,  
                            triangleNormalPoint,    
                            triangleNeighbors,  
                            trianglePointIndices,   
                            thickness,  
                            numberOfLevels, 
                            nTot,
                            betaVolume, 
                            laserParameter, 
                            crystal,    
                            betaCells,  
                            triangleSurfaces,   
                            triangleCenterX,    
                            triangleCenterY,    
                            claddingCellTypes,  
                            claddingNumber, 
                            claddingAbsorption, 
                            refractiveIndices,  
                            reflectivities, 
                            FOLDER):
    
    CURRENT_DIR = subprocess.run("pwd")

    subprocess.run("mkdir "+FOLDER)
    subprocess.run("cd "+FOLDER)

    with open('points.txt','w') as f:
      print(points, file= f)
    

    with open('triangleNormalsX.txt','w') as f:
      print(triangleNormalsX, file= f)
    

    with open('triangleNormalsY.txt','w') as f:
      print(triangleNormalsY, file= f)
    

    with open('forbiddenEdge.txt','w') as f:
      print(forbiddenEdge, file= f)
    

    with open('triangleNormalPoint.txt','w') as f:
      print(triangleNormalPoint, file= f)
    

    with open('triangleNeighbors.txt','w') as f:
      print(triangleNeighbors, file= f)
    

    with open('trianglePointIndices.txt','w') as f:
      print(trianglePointIndices, file= f)
    

    # thickness of one slice!
    with open('thickness.txt','w') as f:
      print(thickness, file= f)
    

    # number of slices
    with open('numberOfLevels.txt','w') as f:
      print(numberOfLevels, file= f)
    

    with open('numberOfTriangles.txt','w') as f:
      a = len(trianglePointIndices, file= f)
      print(a)
    

    with open('numberOfPoints.txt','w') as f:
      a = len(points, file= f)
      print(a)
    

    with open('nTot.txt','w') as f:
      print(nTot, file= f)
    

    with open('betaVolume.txt','w') as f:
      print(betaVolume, file= f)
    

    with open('sigmaA.txt','w') as f:
      print(laserParameter.s_abs, file= f)
    

    with open('sigmaE.txt','w') as f:
      print(laserParameter.s_ems, file= f)
    

    with open('lambdaA.txt','w') as f:
      print(laserParameter.l_abs, file= f)
    

    with open('lambdaE.txt','w') as f:
      print(laserParameter.l_ems, file= f)
    

    with open('crystalTFluo.txt','w') as f:
      print(crystal.tfluo, file= f)
    

    with open('betaCells.txt','w') as f:
      print(betaCells, file= f)
    

    with open('triangleSurfaces.txt','w') as f:
      print(triangleSurfaces, file= f)
    

    with open('triangleCenterX.txt','w') as f:
      print(triangleCenterX, file= f)
    

    with open('triangleCenterY.txt','w') as f:
      print(triangleCenterY, file= f)
    

    with open('claddingCellTypes.txt','w') as f:
      print(claddingCellTypes, file= f)
    

    with open('claddingNumber.txt','w') as f:
      print(claddingNumber, file= f)
    

    with open('claddingAbsorption.txt','w') as f:
      print(claddingAbsorption, file= f)
    

    with open('refractiveIndices.txt','w') as f:
      print(refractiveIndices, file= f)
    

    with open('reflectivities.txt','w') as f:
      print(reflectivities, file= f)
    

    subprocess.run("cd "+CURRENT_DIR)

################################ calcPhiASE ########################################
# % calculates the phiASE values for a given input
# most meshing paramers are given through the function parameters.
# However, many parameters for optimization of the computation are
# set in the beginning of the function (adjust as needed)
# 
# for most mesh parameters see README file
#
# @return phiASE the ASE-Flux for all the given sample points
# @return mseValues the MeanSquaredError values corresponding to phiASE
# @return raysUsedPerSample the number of rays used to calculate each phiASE value
#
####################################################################################

def calcPhiASE(
  points,
  trianglePointIndices,
  betaCells,
  betaVolume,
  claddingCellTypes,
  claddingNumber,
  claddingAbsorption,
  useReflections,
  refractiveIndices,
  reflectivities,
  triangleNormalsX,
  triangleNormalsY,
  triangleNeighbors,
  triangleSurfaces,
  triangleCenterX,
  triangleCenterY,
  triangleNormalPoint,
  forbiddenEdge,
  minRaysPerSample,
  maxRaysPerSample,
  mseThreshold,
  repetitions,
  nTot,
  thickness,
  laserParameter,
  crystal,
  numberOfLevels,
  deviceMode,
  parallelMode,
  maxGPUs,
  nPerNode
  ):

  ############# auto-generating some more input #############
  minSample=0
  [nP,b] = [len(points), len(points)[0]] 
  maxSample=(numberOfLevels*nP)-1

  if useReflections == True:
      REFLECT=' --reflection=1'
  else:
      REFLECT=' --reflection=0'

  if parallelMode=='mpi':
    Prefix=[ 'mpiexec -npernode ', str(nPerNode), ' ' ]
    maxGPUs=1
  else:
    Prefix=''

  # create a tmp-folder in the same directory as this script
  # before filedirectory, name and extension, but only filddirectory is used
  CALCPHIASE_DIR= os.getcwd() 
  TMP_FOLDER= os.path.join(CALCPHIASE_DIR,'input_tmp')

  ################## doing the computation ##################
  # make sure that the temporary folder is clean 
  clean_IO_files(TMP_FOLDER)

  # create new input in the temporary folder #TODO
  create_calcPhiASE_input(points,
                          triangleNormalsX,
                          triangleNormalsY,
                          forbiddenEdge,
                          triangleNormalPoint,
                          triangleNeighbors,
                          trianglePointIndices,
                          thickness,
                          numberOfLevels,
                          nTot,
                          betaVolume,
                          laserParameter,
                          crystal,
                          betaCells,
                          triangleSurfaces,
                          triangleCenterX,
                          triangleCenterY,
                          claddingCellTypes,
                          claddingNumber,
                          claddingAbsorption,
                          refractiveIndices,
                          reflectivities,
                          TMP_FOLDER)

  # do the propagation
  status = subprocess.call(Prefix +CALCPHIASE_DIR+'/bin/calcPhiASE '
                           + '--parallel-mode='+ parallelMode
                           + ' --device-mode='+ deviceMode
                           + ' --min-rays='+ str(minRaysPerSample)
                           + ' --max-rays='+ str(maxRaysPerSample)
                           + REFLECT
                           + ' --input-path='+ TMP_FOLDER
                           + ' --output-path='+ TMP_FOLDER
                           + ' --min-sample-i='+ str(minSample)
                           + ' --max-sample-i='+ str(maxSample)
                           + ' --ngpus='+ str(maxGPUs)
                           + ' --repetitions='+ str(repetitions)
                           + ' --mse-threshold='+ str(mseThreshold)
                           + ' --spectral-resolution='+ str(laserParameter.l_res) )

  if status != 0:
      print('this step of the raytracing computation did NOT finish successfully. Aborting.')

  # get the result
  [ mseValues, raysUsedPerSample, phiASE ] = parse_calcPhiASE_output(TMP_FOLDER)

  # cleanup
  clean_IO_files(TMP_FOLDER)

######################### parse_calcPhiASE_output #############################
#
# takes the output from the CUDA code and fills it into a variable
# assumes that the matrix is saved as a 3D-matrix where the first line 
# denotes the dimensions and the second line is the whole data
#
# @param FOLDER the folder which contains the output files
#
###############################################################################

def parse_calcPhiASE_output (FOLDER):
  CURRENT_DIR = subprocess.call("pwd")
  subprocess.run("cd "+FOLDER)
  with open('phi_ASE.txt','r') as fid:
    arraySize = [int(x) for x in next(fid).split()]
    phiASE = [float(x) for x in next(fid).split()] 
    phiASE = np.reshape(phiASE,arraySize, order= 'F') #order= 'F' beacause file was adapted from Matlab

  with open('mse_values.txt', 'r') as fid: 
    arraySize= [int(x) for x in next(fid).split()]
    mseValues = [float(x) for x in next(fid).split()]
    mseValues = np.reshape(mseValues,arraySize, order= 'F')


  with open('N_rays.txt', 'r') as fid: 
    arraySize = [int(x) for x in next(fid).split()]
    raysUsedPerSample = [float(x) for x in next(fid).split()]
    raysUsedPerSample = np.reshape(raysUsedPerSample,arraySize, order= 'F')

  subprocess.run("cd "+CURRENT_DIR)
  return([mseValues, raysUsedPerSample, phiASE])


