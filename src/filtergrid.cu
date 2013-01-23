#include "filtergrid.h"
#include "geometry.h"

int cellIdx(const Grid *grid, int i, int j, int k){
	return k*grid->dim.x*grid->dim.y+j*grid->dim.x+i;
}

__global__ void filter(const Grid *grid, const RayCu *ray, char *results) {
  int gx = threadIdx.x;
  int gy = threadIdx.y;
  int gz = threadIdx.z;

  PointCu a, b, c, d;
  PrismCu p1, p2;

  VectorCu gridAabbDim = createVector(grid->aabb.min, grid->aabb.max);
  VectorCu cellDim;
  cellDim.x = gridAabbDim.x / grid->dim.x;
  cellDim.y = gridAabbDim.y / grid->dim.y;
  cellDim.z = gridAabbDim.z / grid->dim.z;

  a = createPoint(gx*cellDim.x, gy*cellDim.y, gz*cellDim.z, cellDim.z);
  b = createPoint(gx*cellDim.x, gy*cellDim.y, gz*cellDim.z, cellDim.z);
  c = createPoint(gx*cellDim.x, gy*cellDim.y, gz*cellDim.z, cellDim.z);
  d = createPoint(gx*cellDim.x, gy*cellDim.y, gz*cellDim.z, cellDim.z);

  p1.t1 = createTriangle(a, b, c);
  p2.t1 = createTriangle(b, c, d);

  results[cellIdx(grid, gx, gy, gz)] = (collide_prism_gpu(p1, *ray) != 0) || (collide_prism_gpu(p2, *ray) != 0);
}

//int iIdx(const Grid *grid, float x){
//	assert(x >= grid->aabb.min.x && x <= grid->aabb.max.x);
//	
//	return (x - grid->aabb.min.x)/(grid->aabb.max.x - grid->aabb.min.x) * grid->dim.x;
//}
//
//int jIdx(const Grid *grid, float y){
//	assert(y >= grid->aabb.min.y && y <= grid->aabb.max.y);
//	
//	return (y - grid->aabb.min.y)/(grid->aabb.max.y - grid->aabb.min.y) * grid->dim.y;
//}
//
//int kIdx(const Grid *grid, float z){
//	assert(z >= grid->aabb.min.z && z <= grid->aabb.max.z);
//	
//	return (z - grid->aabb.min.z)/(grid->aabb.max.z - grid->aabb.min.z) * grid->dim.z;
//}
//
//void cellIdx2ijk(const Grid *grid, int cellIdx, int *i, int *j, int *k){
//	
//	int dimXdimY=grid->dim.x*grid->dim.y;
//	int tmp;
//	
//	*k = cellIdx / dimXdimY;
//	
//	tmp=cellIdx - *k * dimXdimY;
//	//*j = (cellIdx - *k * dimXdimY) / grid->dim.x;
//	*j = tmp / grid->dim.x;
//	
//	//*i = cellIdx - *k * dimXdimY - *j* grid->dim.x;
//	*i = tmp - *j * grid->dim.x;
//}


//PointCu* nearest(const Point *p, const PointCu *a, const PointCu *b) {
//  if(a.w == -1) return b;
//  if(b.w == -1) return a;
//
//  if(p < )
//}
//
//PointCu nextIntersection(const RayCu *ray, PointCu *p, int4 *on, int4 dim) {
//  PointCu x, y, z;
//
//  x.w = -1; y.w = -1; z.w = -1; // mark as invalid
//
//  if(!on.x && !on.y && !on.z) { // start position, not on grid
//    gridRound(p.x, dim.x, ray->direction.x);
//  } else if (on.x && !on.y && !on.z) { // on x
//  
//  } else if (!on.x && on.y && !on.z) { // on y
//  
//  } else if (!on.x && !on.y && on.z) { // on z
//  
//  } else if (on.x && on.y && !on.z) { // on x and y
//  
//  } else if (on.x && !on.y && on.z) { // on x and z
//  
//  } else if (!on.x && on.y && on.z) { // on y and z
//
//  } else if (on.x && !on.y && on.z) { // on x, y and z
//  }
//
//  return *nearest(p, nearest(p, x, y), z);
//}
//
//int filter_serial(const Grid *grid, const RayCu *ray, int* results) {
//  PointCu end = addVectorToPoint(ray->P, ray->direction);
//  PointCu p = ray->P;
//  //int4 cellpos;
//  int4 on; // => is p on grid? interesection with grid by x, y and z planes
//
//  //cellpos.x = p.x / grid->dimensions.x;
//  //cellpos.y = p.y / grid->dimensions.y;
//  //cellpos.z = p.z / grid->dimensions.z;
//
//  bool on.x = false, on.y = false, on.z = false;
//  if(p.x % grid->dimensions.x == 0) on.x = true;
//  if(p.y % grid->dimensions.y == 0) on.y = true;
//  if(p.z % grid->dimensions.z == 0) on.z = true;
//
//  nextIntersection(ray, &p, &on, grid->dimensions);
//}