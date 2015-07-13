/*---------------------------------------------------------*/
/*                  N-Body simulation benchmark            */
/*                   written by M.S.Ozhgibesov             */
/*                         04 July 2015                    */
/*---------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <unistd.h>


#define ALLOC    alloc_if(1) free_if(0)
#define FREE     alloc_if(0) free_if(1)
#define REUSE    alloc_if(0) free_if(0)

#define HOSTLEN 50

// Main subroutine for GPU
void cudaDriver(float *rA, float *vA, float *fA, \
                int nBod, int nGpu, int jobId);

__attribute__ ((target(mic))) int numProc;

// Initial conditions
void initCoord(float *rA, float *vA, float *fA, \
               float initDist, int nBod, int nI);

// Forces acting on each body
__attribute__ ((target(mic))) void forcesPhi(float *rA, float *fA, int nBod, int nGpu);

// Calculate velocities and update coordinates
__attribute__ ((target(mic))) void integrationPhi(float *rA, float *vA, float *fA, int nBod, int nGpu);

int main(int argc, const char * argv[]) {
   int const nI = 64;               // Number of bodies in X, Y and Z directions
   int const nBod = nI*nI*nI;       // Total Number of bodies
   int const maxIter = 20;          // Total number of iterations (time steps)
   int const nGpu = nBod/2;
   int const nPhi = nBod - nGpu;
   float const initDist = 1.0;      // Initial distance between the bodies
   float *rA;                       // Coordinates
   float *vA;                       // Velocities
   float *fA;                       // Forces
   int iter;
   int jobId;                       // GPU job ID (0 - allocation and copy of initial data)
                                    // (1 - simulation; 2 - deallocation and copy out)
   double startTime0, endTime0;
   char host[HOSTLEN];

   rA = (float*)malloc(3*nBod*sizeof(float));
   fA = (float*)malloc(3*nBod*sizeof(float));
   vA = (float*)malloc(3*nBod*sizeof(float));

   #pragma offload target(mic:0) out(numProc, host)
   {
      gethostname(host, HOSTLEN);
      numProc = omp_get_num_procs();
   }
   printf("MIC name: %s\n", host);

   // Setup initial conditions
   initCoord(rA, vA, fA, initDist, nBod, nI);

   jobId = 0;
   // Copy initial data to GPU
   cudaDriver(rA, vA, fA, nBod, nGpu, jobId);

   printf("nGpu = %d",nGpu);
   // Copy initial data to Xeon Phi
   #pragma offload_transfer target(mic:0) in(rA, fA, vA:length(3*nBod) ALLOC) in(nBod, nGpu)

   startTime0 = omp_get_wtime();
   // Main loop
   for ( iter = 0; iter < maxIter; iter++ ) {

      #pragma omp parallel num_threads(2)
      #pragma omp sections
      {
         #pragma omp section
            #pragma offload target(mic:0) inout(rA:length(3*nBod)) \
               nocopy(fA, vA:length(3*nBod) REUSE)
            {
               forcesPhi(rA, fA, nBod, nGpu);
               integrationPhi(rA, vA, fA, nBod, nGpu);
            }
         #pragma omp section
         {
            jobId = 1;
            cudaDriver(rA, vA, fA, nBod, nGpu, jobId);
         }
      }
      jobId = 2;
      cudaDriver(rA, vA, fA, nBod, nGpu, jobId);
   }
   endTime0 = omp_get_wtime();

   printf("\nTotal time = %10.4f [sec]\n", endTime0 - startTime0);


   // Free memory on devices
   jobId = 3;
   cudaDriver(rA, vA, fA, nBod, nGpu, jobId);
   #pragma offload_transfer target(mic:0) in(rA, fA, vA:length(3*nBod) FREE)

   free(rA);
   free(vA);
   free(fA);
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
__attribute__ ((target(mic))) void forcesPhi(float *rA, float *fA, int nBod, int nGpu)
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

   #pragma omp parallel for num_threads(numProc) private(Xi, Yi, Zi, \
               Xij, Yij, Zij, magForce, invRij2, invRij6, j, i)
   for (i = nGpu; i < nBod; i++) {
      Xi = rAx[i];
      Yi = rAy[i];
      Zi = rAz[i];
      fAx[i] = 0.f;
      fAy[i] = 0.f;
      fAz[i] = 0.f;
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
}

// Integration of coordinates an velocities
__attribute__ ((target(mic))) void integrationPhi(float *rA, float *vA, float *fA, \
                                                  int nBod, int nGpu)
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

   #pragma omp parallel for num_threads(numProc) private(i)
   for (i = nGpu; i < nBod; i++) {
      rAx[i]+= (vAx[i] + fAx[i]*mdthalf)*dt;
      rAy[i]+= (vAy[i] + fAy[i]*mdthalf)*dt;
      rAz[i]+= (vAz[i] + fAz[i]*mdthalf)*dt;

      vAx[i]+= fAx[i]*dt;
      vAy[i]+= fAy[i]*dt;
      vAz[i]+= fAz[i]*dt;
   }
}
