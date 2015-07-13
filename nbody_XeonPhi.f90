   program nbody_XeonPhi
      use omp_lib
      implicit none
      integer, parameter:: nI = 32        ! Number of bodies in X, Y and Z directions  
      integer, parameter:: nBod = nI**3   ! Total Number of bodies
      integer, parameter:: maxIter = 20   ! Total number of iterations (time steps)
      integer:: numProc
      integer:: iter
      character(len=50):: host
      real(4), parameter:: initDist = 1.0 ! Initial distance between the bodies
      real(4), allocatable:: rA(:)        ! Coordinates
      real(4), allocatable:: vA(:)        ! Velocities
      real(4), allocatable:: fA(:)        ! Forces
      real(8):: startTime0, endTime0
      common/ourCommonData/numProc



      allocate(rA(3*nBod), vA(3*nBod), fA(3*nBod))

      ! Mark variable numProc as needing to be allocated
      ! on both the host and device
      !DIR$ ATTRIBUTES OFFLOAD:mic::numProc, hostnm
      !DIR$ OFFLOAD BEGIN TARGET(mic) OUT(host, numProc)
      call hostnm(host) 
      numProc = omp_get_num_procs()
      !DIR$ END OFFLOAD

      write(*,'(A11,A50)')"Host name: ", host
      write(*,'(A32,I4)')"Available number of processors: ",numProc

      ! Setup initial conditions
      call initCoord(rA, vA, fA, initDist, nBod, nI)

      ! Mark routines integration and forces as needing both
      ! host and coprocessor version
      !DIR$ ATTRIBUTES OFFLOAD:mic::integration, forces

      ! Main loop
      startTime0 = omp_get_wtime()
      !DIR$ OFFLOAD BEGIN TARGET(mic) INOUT(rA,fA,vA:length(3*nBod))
      do iter = 1, maxIter
         call forces(rA, vA, nBod)

         call integration(rA, vA, fA, nBod)
      enddo
      !DIR$ END OFFLOAD
      endTime0 = omp_get_wtime()

      write(*,'(A13,F10.4,A6)'), "Total time = ", endTime0 - startTime0," [sec]"

      deallocate(rA, vA, fA)
   end program nbody_XeonPhi


   ! Initial conditions
   subroutine initCoord(rA, vA, fA, initDist, nBod, nI)
      implicit none
      integer:: i, j, k, ii
      integer:: nI, nBod
      integer:: initDist
      integer:: numProc
      real(4):: Xi, Yi,Zi
      real(4):: rA(*), fA(*), vA(*)

      fA(1:3*nBod) = 0.D0
      vA(1:3*nBod) = 0.D0
      ii = 1
      do i = 1, nI
         Xi = i*(initDist - 1)
         do j = 1, nI
            Yi = j*(initDist - 1)
            do k = 1, nI
               Zi = k*(initDist - 1)
               rA(ii       ) = Xi
               rA(ii+  nBod) = Yi
               rA(ii+2*nBod) = Zi
               ii = ii + 1
            enddo
         enddo
      enddo

   end subroutine initCoord

   ! Forces acting on each body
   !DIR$ ATTRIBUTES OFFLOAD:mic:: forces
   subroutine forces(rA, fA, nBod)
      implicit none
      integer:: i, j
      integer:: nI, nBod
      integer:: numProc
      real(4):: Xi, Yi, Zi
      real(4):: Xij, Yij, Zij             ! X[j] - X[i] and so on
      real(4):: Rij2                      ! Xij^2+Yij^2+Zij^2
      real(4):: invRij2, invRij6          ! 1/rij^2; 1/rij^6
      real(4):: rA(*), fA(*)
      real(4):: magForce                  ! Force magnitude
      real(4):: fAix, fAiy, fAiz
      real(4), parameter:: EPS = 1.E-10   ! Small value to prevent 0/0 if i==j
      common/ourCommonData/numProc

      !$OMP PARALLEL NUM_THREADS(numProc) &
      !$OMP PRIVATE(Xi, Yi, Zi, Xij, Yij, Zij, magForce, invRij2, invRij6, i, j)&
      !$OMP PRIVATE(fAix, fAiy, fAiz)
      !$OMP DO
      do i = 1, nBod
         Xi = rA(i       )
         Yi = rA(i+  nBod)
         Zi = rA(i+2*nBod)
         fAix = 0.E0
         fAiy = 0.E0
         fAiz = 0.E0
         do j = 1, nBod
            Xij = rA(j       ) - Xi
            Yij = rA(j+  nBod) - Yi
            Zij = rA(j+2*nBod) - Zi
            Rij2 = Xij*Xij + Yij*Yij + Zij*Zij
            invRij2 = Rij2/((Rij2 + EPS)**2)
            invRij6 = invRij2*invRij2*invRij2
            magForce = 6.0*invRij2*(2.0*invRij6 - 1.0)*invRij6
            fAix = fAix + Xij*magForce
            fAiy = fAiy + Yij*magForce
            fAiz = fAiz + Zij*magForce
         enddo
         fA(i       ) = fAix
         fA(i+  nBod) = fAiy
         fA(i+2*nBod) = fAiz
      enddo
      !$OMP END PARALLEL

   end subroutine forces

   !DIR$ ATTRIBUTES OFFLOAD:mic::integration
   subroutine integration(rA, vA, fA, nBod)
      implicit none
      integer:: i
      integer:: nI, nBod
      integer:: numProc
      real(4), parameter:: dt = 0.01            ! Time step
      real(4), parameter:: mass = 1.0           ! mass of a body
      real(4), parameter:: mdthalf = dt*0.5/mass
      real(4):: rA(*), vA(*), fA(*)
      common/ourCommonData/numProc

      !$OMP PARALLEL NUM_THREADS(numProc) PRIVATE(i)
      !$OMP DO
      do i = 1, 3*nBod
         rA(i) = (rA(i) + fA(i)*mdthalf)*dt
         vA(i) = fA(i)*dt
      enddo
      !$OMP END PARALLEL

   end subroutine integration
