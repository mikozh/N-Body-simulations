#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cuda.h>


// Forces acting on each body
__global__ void forcesGPU(float *rA, float *fA, int nBod);

// Calculate velocities and update coordinates
__global__ void integrationGPU(float *rA, float *vA, float *fA, int nBod);

extern "C" void cudaDriver(float *rA, float *vA, float *fA, int nBod, int nGpu, int jobId) {
   float static *rA_d;                // Coordinates
   float static *vA_d;                // Velocities
   float static *fA_d;                // Forces
   float time;
   float static totalTime = 0.f;
   int static nPhi = nBod - nGpu;

   cudaDeviceProp devProp;
   cudaEvent_t start, stop;

   if ( jobId == 0) {
      cudaMalloc((void**)&rA_d, 3*nBod*sizeof(float));
      cudaMalloc((void**)&vA_d, 3*nBod*sizeof(float));
      cudaMalloc((void**)&fA_d, 3*nBod*sizeof(float));

      cudaGetDeviceProperties(&devProp, 0);
      printf("Name of CUDA GPU: %s\n",devProp.name);

      cudaMemcpy(rA_d, rA, 3*nBod*sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(vA_d, vA, 3*nBod*sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(fA_d, fA, 3*nBod*sizeof(float), cudaMemcpyHostToDevice);
   }

   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   // Run calculations and copy out the results
   if ( jobId == 1) {
      cudaEventRecord(start, 0);

      forcesGPU<<<nGpu/512, 512>>>(rA_d, fA_d, nBod);

      integrationGPU<<<nGpu/512, 512>>>(rA_d, vA_d, fA_d, nBod);

      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
   }

   // Get coordinates computed by Xeon Phi
   if ( jobId == 2) {
      cudaMemcpy(&rA[     0], &rA_d[     0], nGpu*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(&rA[  nBod], &rA_d[  nBod], nGpu*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(&rA[2*nBod], &rA_d[2*nBod], nGpu*sizeof(float), cudaMemcpyDeviceToHost);

      cudaEventRecord(start, 0);
      cudaMemcpy(&rA_d[nGpu       ], &rA[nGpu       ], nPhi*sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(&rA_d[nGpu+  nBod], &rA[nGpu+  nBod], nPhi*sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(&rA_d[nGpu+2*nBod], &rA[nGpu+2*nBod], nPhi*sizeof(float), cudaMemcpyHostToDevice);
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
   }


   cudaEventElapsedTime(&time, start, stop);
   totalTime+= time;


   if ( jobId == 3) {
      cudaMemcpy(&rA[     0], &rA_d[     0], nGpu*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(&rA[  nBod], &rA_d[  nBod], nGpu*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(&rA[2*nBod], &rA_d[2*nBod], nGpu*sizeof(float), cudaMemcpyDeviceToHost);

      cudaMemcpy(&vA[     0], &vA_d[     0], nGpu*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(&vA[  nBod], &vA_d[  nBod], nGpu*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(&vA[2*nBod], &vA_d[2*nBod], nGpu*sizeof(float), cudaMemcpyDeviceToHost);

      cudaMemcpy(&fA[     0], &fA_d[     0], nGpu*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(&fA[  nBod], &fA_d[  nBod], nGpu*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(&fA[2*nBod], &fA_d[2*nBod], nGpu*sizeof(float), cudaMemcpyDeviceToHost);

      printf("\nTotal GPU time = %10.4f [sec]\n", totalTime*1.E-3);

      cudaFree(rA_d);
      cudaFree(vA_d);
      cudaFree(fA_d);
   }
}

// Forces acting on each body
__global__ void forcesGPU(float *rA, float *fA, int nBod)
{
   int i, j, k;
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
   float fAxi, fAyi, fAzi;
   float __shared__ rxj[512];
   float __shared__ ryj[512];
   float __shared__ rzj[512];

   i = blockDim.x*blockIdx.x + threadIdx.x;
   Xi = rAx[i];
   Yi = rAy[i];
   Zi = rAz[i];
   fAxi = 0.0;
   fAyi = 0.0;
   fAzi = 0.0;
   for (k = 0; k < nBod; k+=512) {
      rxj[threadIdx.x] = rAx[k+threadIdx.x];
      ryj[threadIdx.x] = rAy[k+threadIdx.x];
      rzj[threadIdx.x] = rAz[k+threadIdx.x];
      syncthreads();

      for (j = 0; j < 512; j++) {
         Xij = rxj[j] - Xi;
         Yij = ryj[j] - Yi;
         Zij = rzj[j] - Zi;
         Rij2 = Xij*Xij + Yij*Yij + Zij*Zij;
         invRij2 = Rij2/((Rij2 + EPS)*(Rij2 + EPS));
         invRij6 = invRij2*invRij2*invRij2;
         magForce = 6.f*invRij2*(2.f*invRij6 - 1.f)*invRij6;
         fAxi+= Xij*magForce;
         fAyi+= Yij*magForce;
         fAzi+= Zij*magForce;
      }
   }
   fAx[i] = fAxi;
   fAy[i] = fAyi;
   fAz[i] = fAzi;
}

// Integration of coordinates an velocities
__global__ void integrationGPU(float *rA, float *vA, float *fA, int nBod)
{
   int i;
   float const dt = 0.01;              // Time step
   float const mass = 1.0;             // mass of a body
   float const mdthalf = dt*0.5/mass;
   float *rAx = &rA[     0];
   float *rAy = &rA[  nBod];
   float *rAz = &rA[2*nBod];
   float *vAx = &vA[     0];
   float *vAy = &vA[  nBod];
   float *vAz = &vA[2*nBod];
   float *fAx = &fA[     0];
   float *fAy = &fA[  nBod];
   float *fAz = &fA[2*nBod];

   i = blockDim.x*blockIdx.x + threadIdx.x;

   rAx[i]+= (vAx[i] + fAx[i]*mdthalf)*dt;
   rAy[i]+= (vAy[i] + fAy[i]*mdthalf)*dt;
   rAz[i]+= (vAz[i] + fAz[i]*mdthalf)*dt;

   vAx[i]+= fAx[i]*dt;
   vAy[i]+= fAy[i]*dt;
   vAz[i]+= fAz[i]*dt;
}
