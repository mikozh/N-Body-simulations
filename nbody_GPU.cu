#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cuda.h>

#define HOSTLEN 50


// Initial conditions
void initCoord(float *rA, float *vA, float *fA, \
               float initDist, int nBod, int nI);

// Forces acting on each body
__global__ void forces(float *rA, float *fA, int nBod);

// Calculate velocities and update coordinates
__global__ void integration(float *rA, float *vA, float *fA, int nBod);

int main(int argc, const char * argv[]) {
   int const nI = 32;               // Number of bodies in X, Y and Z directions
   int const nBod = nI*nI*nI;       // Total Number of bodies
   int const maxIter = 20;          // Total number of iterations (time steps)
   float const initDist = 1.0;      // Initial distance between the bodies
   float *rA, *rA_d;                // Coordinates
   float *vA, *vA_d;                // Velocities
   float *fA, *fA_d;                // Forces
   float time;
   int iter;
   cudaDeviceProp devProp;
   cudaEvent_t start, stop;

   rA = (float*)malloc(3*nBod*sizeof(float));
   fA = (float*)malloc(3*nBod*sizeof(float));
   vA = (float*)malloc(3*nBod*sizeof(float));

   cudaMalloc((void**)&rA_d, 3*nBod*sizeof(float));
   cudaMalloc((void**)&vA_d, 3*nBod*sizeof(float));
   cudaMalloc((void**)&fA_d, 3*nBod*sizeof(float));

   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   cudaGetDeviceProperties(&devProp, 0);
   printf("Name of CUDA GPU: %s\n",devProp.name);

   // Setup initial conditions
   initCoord(rA, vA, fA, initDist, nBod, nI);

   cudaEventRecord(start, 0);

   cudaMemcpy(rA_d, rA, 3*nBod*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(vA_d, vA, 3*nBod*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(fA_d, fA, 3*nBod*sizeof(float), cudaMemcpyHostToDevice);

   // Main loop
   for ( iter = 0; iter < maxIter; iter++ ) {
      forces<<<nBod/512, 512>>>(rA_d, fA_d, nBod);

      integration<<<3*nBod/512, 512>>>(rA_d, vA_d, fA_d, nBod);
   }

   cudaMemcpy(rA, rA_d, 3*nBod*sizeof(float), cudaMemcpyDeviceToHost);
   cudaMemcpy(vA, vA_d, 3*nBod*sizeof(float), cudaMemcpyDeviceToHost);
   cudaMemcpy(fA, fA_d, 3*nBod*sizeof(float), cudaMemcpyDeviceToHost);

   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);

   cudaEventElapsedTime(&time, start, stop);
   printf("\nTotal time = %10.4f [sec]\n", time*1.E-3);

   free(rA);
   free(vA);
   free(fA);

   cudaFree(rA_d);
   cudaFree(vA_d);
   cudaFree(fA_d);
   return 0;
}

// Initial conditions
void initCoord(float *rA, float *vA, float *fA, \
               float initDist, int nBod, int nI)
{
   int i, j, k;
   float Xi, Yi, Zi;
   float *rAx = &rA[     0];        //----
   float *rAy = &rA[  nBod];        // Pointers on X, Y, Z components of coordinates
   float *rAz = &rA[2*nBod];        //----
   int ii = 0;

   memset(fA, 0.0, 3*nBod*sizeof(float));
   memset(vA, 0.0, 3*nBod*sizeof(float));

   for (i = 0; i < nI; i++) {
      Xi = i*initDist;
      for (j = 0; j < nI; j++) {
         Yi = j*initDist;
         for (k = 0; k < nI; k++) {
            Zi = k*initDist;
            rAx[ii] = Xi;
            rAy[ii] = Yi;
            rAz[ii] = Zi;
            ii++;
         }
      }
   }
}

// Forces acting on each body
__global__ void forces(float *rA, float *fA, int nBod)
{
   int i, j;
   float Xi, Yi, Zi;
   float Xij, Yij, Zij;             // X[j] - X[i] and so on
   float Rij2;                      // Xij^2+Yij^2+Zij^2
   float invRij2, invRij6;          // 1/rij^2; 1/rij^6
   float *rAx = &rA[     0];        //----
   float *rAy = &rA[  nBod];        // Pointers on X, Y, Z components of coordinates
   float *rAz = &rA[2*nBod];        //----
   float *fAx = &fA[     0];        //----
   float *fAy = &fA[  nBod];        // Pointers on X, Y, Z components of forces
   float *fAz = &fA[2*nBod];        //----
   float magForce;                  // Force magnitude
   float const EPS = 1.E-10;        // Small value to prevent 0/0 if i==j

   i = blockDim.x*blockIdx.x + threadIdx.x;
   Xi = rAx[i];
   Yi = rAy[i];
   Zi = rAz[i];
   fAx[i] = 0.0;
   fAy[i] = 0.0;
   fAz[i] = 0.0;
   for (j = 0; j < nBod; j++) {
      Xij = rAx[j] - Xi;
      Yij = rAy[j] - Yi;
      Zij = rAz[j] - Zi;
      Rij2 = Xij*Xij + Yij*Yij + Zij*Zij;
      invRij2 = Rij2/((Rij2 + EPS)*(Rij2 + EPS));
      invRij6 = invRij2*invRij2*invRij2;
      magForce = 6.f*invRij2*(2.f*invRij6 - 1.f)*invRij6;
      fAx[i]+= Xij*magForce;
      fAy[i]+= Yij*magForce;
      fAz[i]+= Zij*magForce;
   }
}

// Integration of coordinates an velocities
__global__ void integration(float *rA, float *vA, float *fA, int nBod)
{
   int i;
   float const dt = 0.01;              // Time step
   float const mass = 1.0;             // mass of a body
   float const mdthalf = dt*0.5/mass;

   i = blockDim.x*blockIdx.x + threadIdx.x;

   rA[i]+= (vA[i] + fA[i]*mdthalf)*dt;

   vA[i]+= fA[i]*dt;
}
