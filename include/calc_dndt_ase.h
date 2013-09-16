#ifndef RAY_PROPAGATION_GPU_KERNEL_H
#define RAY_PROPAGATION_GPU_KERNEL_H
#include <vector>
#include <mesh.h>

/**
 * @brief Calculates the integral over time of ASE for each 
 *        point in the given mesh. The calculation is
 *        accelerated by CUDA (compute capability 2.0).
 *
 * @author Erik Zenker
 * @author Carlchristian Eckert
 * @author Marius Melzer
 *
 * @licence GPLv3
 *
 * @param &threads Number of threads used by kernel.
 *                 By reference because it can be used by calling
 *                 function for statistics etc.
 *
 * @param &blocks Number of blocks used by kernel is 
 *                maximal 200 because of mersenne twister.
 *                By reference because it can be used by calling
 *                function for statistics etc.
 *
 * @param &hostRaysPerSample  Number of rays propagated for each sample point.
 *                            By reference because it can be used by calling
 *                            function for statistics etc.
 *
 * @param mesh All information about triangles, points, contants. 
 *             See mesh.h for details.
 *
 * @param sigmaA
 *
 * @param sigmaE
 *
 * @param dndtAse Solution vector with dndt ASE values
 *
 * @return runtime time comsumption of core algorithm (mostly gpu)         
 *
 **/
float calcDndtAse (unsigned &threads, 
		   unsigned &blocks, 
		   unsigned &hostRaysPerSample,
		   unsigned maxRaysPerSample,
		   Mesh mesh,
		   Mesh hostMesh,
		   std::vector<double> sigmaA,
		   std::vector<double> sigmaE,
		   float expectationThreshold,
		   bool useReflections,
		   std::vector<double> &dndtAse,
		   std::vector<float> &phiAse,
		   std::vector<double> &expectation);


#endif
