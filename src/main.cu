// Libraries
#include <stdio.h> /* fprintf, memcpy, strstr, strcmp */
#include <assert.h> /* assert */
#include <string> /* string */
#include <vector> /* vector */
#include <stdlib.h> /* atoi */
#include <pthread.h> /* pthread_t, pthread_join */
#include <algorithm> /* max_element */

// User header files
#include <calc_phi_ase.h>
#include <parser.h>
#include <write_to_vtk.h>
#include <write_matlab_output.h>
#include <for_loops_clad.h>
#include <cudachecks.h>
#include <mesh.h>
#include <test_environment.h>
#include <thread.h>

#define MIN_COMPUTE_CAPABILITY_MAJOR 2
#define MIN_COMPUTE_CAPABILITY_MINOR 0

/** 
 * @brief Queries for devices on the running mashine and collects
 *        them on the devices array. Set the first device in this 
 *        array as computaion-device. On Errors the programm will
 *        be stoped by exit(). Otherwise you can set the device by command
 *        line parameter --device=
 * 
 * @param verbose > 0 prints debug output
 * 
 * @return vector of possible devices
 */
std::vector<unsigned> getCorrectDevice(int verbose){
  cudaDeviceProp prop;
  int minMajor = MIN_COMPUTE_CAPABILITY_MAJOR;
  int minMinor = MIN_COMPUTE_CAPABILITY_MINOR;
  int count;
  std::vector<unsigned> devices;

  // Get number of devices
  CUDA_CHECK_RETURN( cudaGetDeviceCount(&count));

  // Check devices for compute capability and if device is busy
  for(int i=0; i<count; ++i){
    CUDA_CHECK_RETURN( cudaGetDeviceProperties(&prop, i) );
    if( (prop.major > minMajor) || (prop.major == minMajor && prop.minor >= minMinor) ){
      cudaSetDevice(i);
      int* test;
      if(cudaMalloc((void**) &test, sizeof(int)) == cudaSuccess){
        devices.push_back(i);
        cudaFree(test);
        cudaDeviceReset();
      }
    }
  }

  if(devices.size() == 0){
    fprintf(stderr,"\nNone of the free CUDA-capable devices is sufficient!\n");
    exit(1);
  }

  cudaSetDevice(devices.at(0));

  if(verbose > 0){
    fprintf(stderr,"\nFound %d available CUDA devices with Compute Capability >= %d.%d):\n", int(devices.size()), minMajor,minMinor);
    for(unsigned i=0; i<devices.size(); ++i){
      CUDA_CHECK_RETURN( cudaGetDeviceProperties(&prop, devices[i]) );
      fprintf(stderr,"[%d] %s (Compute Capability %d.%d)\n", devices[i], prop.name, prop.major, prop.minor);
    }
  }

  return devices;

}

double calcDndtAse(const Mesh& mesh, const double sigmaA, const double sigmaE, const float phiAse, const unsigned sample_i){
  double gain_local = mesh.nTot * mesh.betaCells[sample_i] * (sigmaE + sigmaA) - double(mesh.nTot * sigmaA);
  return gain_local * phiAse / mesh.crystalFluorescence;
}

int main(int argc, char **argv){
  unsigned raysPerSample = 0;
  unsigned maxRaysPerSample = 0;
  float maxExpectation = 0;
  float  avgExpectation = 0;
  unsigned highExpectation = 0;
  std::string runmode("");
  std::string compareLocation("");
  float runtime = 0.0;
  bool silent = false;
  bool writeVtk = false;
  bool useReflections = false;
  std::vector<unsigned> devices; // will be assigned in getCOrrectDevice();
  unsigned maxGpus = 0;
  RunMode mode = NONE;

  std::string experimentPath;

  // Wavelength data
  std::vector<double> sigmaA;
  std::vector<double> sigmaE;
  std::vector<float> mseThreshold;

  // Set/Test device to run experiment with
  devices = getCorrectDevice(1);

  // Parse Commandline
  parseCommandLine(argc, argv, &raysPerSample, &maxRaysPerSample, &experimentPath, &silent,
		   &writeVtk, &compareLocation, &mode, &useReflections, &maxGpus);

  // sanity checks
  if(checkParameterValidity(argc, raysPerSample, &maxRaysPerSample, experimentPath, devices.size(), mode, &maxGpus)) return 1;

  // Parse wavelengths from files
  if(fileToVector(experimentPath + "sigma_a.txt", &sigmaA)) return 1;
  if(fileToVector(experimentPath + "sigma_e.txt", &sigmaE)) return 1;
  if(fileToVector(experimentPath + "mse_threshold.txt", &mseThreshold)) return 1;
  assert(sigmaA.size() == sigmaE.size());
  assert(mseThreshold.size() == sigmaE.size());
 

  // Parse experientdata and fill mesh
  Mesh hMesh;
  std::vector<Mesh> dMesh(maxGpus);
  
  if(Mesh::parseMultiGPU(hMesh, dMesh, experimentPath, devices, maxGpus)) return 1;

  // Solution vector
  std::vector<double> dndtAse(hMesh.numberOfSamples * sigmaE.size(), 0);
  std::vector<float>  phiAse(hMesh.numberOfSamples * sigmaE.size(), 0);
  std::vector<double> mse(hMesh.numberOfSamples * sigmaE.size(), 1000);
  std::vector<unsigned> totalRays(hMesh.numberOfSamples * sigmaE.size(), 0);
  
  // Run Experiment
  std::vector<float> runtimes(maxGpus, 0);
  std::vector<pthread_t> threadIds(maxGpus, 0);
  float samplePerGpu = hMesh.numberOfSamples / (float) maxGpus;
  switch(mode){
    case RAY_PROPAGATION_GPU:
      for(unsigned gpu_i = 0; gpu_i < maxGpus; ++gpu_i){
	unsigned minSample_i = gpu_i * samplePerGpu;
	unsigned maxSample_i = min((float)hMesh.numberOfSamples, (gpu_i + 1) * samplePerGpu);
	threadIds[gpu_i] = calcPhiAseThreaded( raysPerSample,
					       maxRaysPerSample,
					       dMesh.at(devices.at(gpu_i)),
					       hMesh,
					       sigmaA,
					       sigmaE,
					       mseThreshold,
					       useReflections,
					       phiAse,
					       mse,
					       totalRays,
					       devices.at(gpu_i),
					       minSample_i,
					       maxSample_i,
					       runtimes.at(gpu_i)
					       );

      }
      joinAll(threadIds);
      for(std::vector<float>::iterator it = runtimes.begin(); it != runtimes.end(); ++it){
	runtime = max(*it, runtime);
      }
      cudaDeviceReset();      
      runmode="Ray Propagation New GPU";
      break;

    case FOR_LOOPS:
      runtime = forLoopsClad( &dndtAse,
			      raysPerSample,
			      &hMesh,
			      hMesh.betaCells,
			      hMesh.nTot,
			      sigmaA.at(0),
			      sigmaE.at(0),
			      hMesh.numberOfPoints,
			      hMesh.numberOfTriangles,
			      hMesh.numberOfLevels,
			      hMesh.thickness,
			      hMesh.crystalFluorescence);
      runmode = "For Loops";
      break;

  case TEST:
    testEnvironment(raysPerSample,
		    maxRaysPerSample,
		    dMesh.at(0),
		    hMesh,
		    sigmaA,
		    sigmaE,
		    mseThreshold.at(0),
		    useReflections,
		    dndtAse,
		    phiAse,
		    mse
		    );
    cudaDeviceReset();
    runmode="Test Environment";
    break;
  default:
    exit(0);
  }

  // Filter maxExpectation
  for(std::vector<double>::iterator it = mse.begin(); it != mse.end(); ++it){
    maxExpectation = max(maxExpectation, *it);
    avgExpectation += *it;
    if(*it > mseThreshold.at(0))
      highExpectation++;
  }
  avgExpectation /= mse.size();


  // Print Solutions
  for(unsigned wave_i = 0; wave_i < sigmaE.size(); ++wave_i){
    fprintf(stderr, "\n\nC Solutions %d\n", wave_i);
    for(unsigned sample_i = 0; sample_i < hMesh.numberOfSamples; ++sample_i){
      int sampleOffset = sample_i + hMesh.numberOfSamples * wave_i;
      dndtAse.at(sampleOffset) = calcDndtAse(hMesh, sigmaA.at(wave_i), sigmaE.at(wave_i), phiAse.at(sampleOffset), sample_i);
      if(silent && sample_i <=10)
	fprintf(stderr, "C Dndt ASE[%d]: %.80f %.10f\n", sample_i, dndtAse.at(sampleOffset), mse.at(sampleOffset));
    }
    for(unsigned sample_i = 0; sample_i < hMesh.numberOfSamples; ++sample_i){
      int sampleOffset = sample_i + hMesh.numberOfSamples * wave_i;
      fprintf(stderr, "C PHI ASE[%d]: %.80f %.10f\n", sample_i, phiAse.at(sampleOffset), mse.at(sampleOffset));
      if(silent){
        if(sample_i >= 10) break;
      }
	}
  }

  // Compare with vtk
  if(compareLocation!="") {
    std::vector<double> compareAse = compareVtk(dndtAse, compareLocation, hMesh.numberOfSamples);

  }

  // Print statistics
  fprintf(stderr, "\n");
  fprintf(stderr, "C Statistics\n");
  fprintf(stderr, "C Prism             : %d\n", (int) hMesh.numberOfPrisms);
  fprintf(stderr, "C Samples           : %d\n", (int) dndtAse.size());
  fprintf(stderr, "C MSE threshold     : %f\n", *(std::max_element(mseThreshold.begin(),mseThreshold.end())));
  fprintf(stderr, "C max. MSE          : %f\n", maxExpectation);
  fprintf(stderr, "C avg. MSE          : %f\n", avgExpectation);
  fprintf(stderr, "C too high MSE      : %d\n", highExpectation);
  fprintf(stderr, "C Runmode           : %s \n", runmode.c_str());
  fprintf(stderr, "C Nr of GPUs        : %d \n", maxGpus);
  fprintf(stderr, "C Runtime           : %f s\n", runtime);
  fprintf(stderr, "\n");

  // Write experiment data
  writeMatlabOutput(
		  experimentPath,
		  phiAse,
		  totalRays,
		  mse,
		  sigmaE.size(),
		  hMesh.numberOfSamples,
		  hMesh.numberOfLevels
      );

  // FOR OUTPUT
  std::vector<double> tmpPhiAse(phiAse.begin(), phiAse.end());
  std::vector<double> tmpTotalRays(totalRays.begin(), totalRays.end());


  //writeVectorToFile(mse, "octrace_expvec");
  if(writeVtk) writeToVtk(hMesh, dndtAse, "octrace_dndt", raysPerSample, maxRaysPerSample, mseThreshold.at(0), useReflections, runtime);
  if(writeVtk) writeToVtk(hMesh, tmpPhiAse, "octrace_phiase", raysPerSample, maxRaysPerSample, mseThreshold.at(0), useReflections, runtime);
  if(writeVtk) writeToVtk(hMesh, mse, "octrace_mse", raysPerSample, maxRaysPerSample, mseThreshold.at(0), useReflections, runtime);
  if(writeVtk) writeToVtk(hMesh, tmpTotalRays, "octrace_total_rays", raysPerSample, maxRaysPerSample, mseThreshold.at(0), useReflections, runtime);
  
  return 0;
}


