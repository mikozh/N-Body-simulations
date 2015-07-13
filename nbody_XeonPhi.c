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

#define HOSTLEN 50

__attribute__ ((target(mic))) int numProc;

// Initial conditions
void initCoord(float *rA, float *vA, float *fA, \
               float initDist, int nBod, int nI);

// Forces acting on each body
__attribute__ ((target(mic))) void forces(float *rA, float *fA, int nBod);

// Calculate velocities and update coordinates
__attribute__ ((target(mic))) void integration(float *rA, float *vA, float *fA, int nBod);

int main(int argc, const char * argv[]) {
   int const nI = 64;               // Number of bodies in X, Y and Z directions
   int const nBod = nI*nI*nI;       // Total Number of bodies
   int const maxIter = 20;          // Total number of iterations (time steps)
   float const initDist = 1.0;      // Initial distance between the bodies
   float *rA;                       // Coordinates
   float *vA;                       // Velocities
   float *fA;                       // Forces
   int iter;
   double startTime0, endTime0;
   double startTime1, endTime1;
   char host[HOSTLEN];

   rA = (float*)malloc(3*nBod*sizeof(float));
   fA = (float*)malloc(3*nBod*sizeof(float));
   vA = (float*)malloc(3*nBod*sizeof(float));

   #pragma offload target(mic) out(numProc, host)
   {
      gethostname(host, HOSTLEN);
      numProc = omp_get_num_procs();
   }
   printf("Host name: %s\n", host);
   printf("Available number of processors: %d\n", numProc);

   // Setup initial conditions
   initCoord(rA, vA, fA, initDist, nBod, nI);

   startTime0 = omp_get_wtime();
   // Main loop
   #pragma offload target(mic) inout(rA, fA, vA:length(3*nBod)) in(nBod)
   for ( iter = 0; iter < maxIter; iter++ ) {
      forces(rA, fA, nBod);
      integration(rA, vA, fA, nBod);
   }
   endTime0 = omp_get_wtime();

   printf("\nTotal time = %10.4f [sec]\n", endTime0 - startTime0);

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
__attribute__ ((target(mic))) void forces(float *rA, float *fA, int nBod)
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
   for (i = 0; i < nBod; i++) {
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
__attribute__ ((target(mic))) void integration(float *rA, float *vA, float *fA, int nBod)
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
   for (i = 0; i < nBod; i++) {
      rAx[i]+= (vAx[i] + fAx[i]*mdthalf)*dt;
      rAy[i]+= (vAy[i] + fAy[i]*mdthalf)*dt;
      rAz[i]+= (vAz[i] + fAz[i]*mdthalf)*dt;

      vAx[i]+= fAx[i]*dt;
      vAy[i]+= fAy[i]*dt;
      vAz[i]+= fAz[i]*dt;
   }
}
