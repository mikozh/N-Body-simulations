N-Body simulations using various parallel programming technologies (CUDA, OpenMP)
and three different target devices (CPU Intel 2680v3, Intel Xeon Phi coprocessor 5110p, 
CUDA GPU Tesla k40m).

Foder PhiGPU contains hybrid code for N-body simulations, i.e. job is shared among 
Xeon Phi coprocessor and GPU.
Timing(262144 bodies, 20 time steps):
GPU: 37.3370sec
MIC: 60.4038sec
Hybrid (half job on MIC, half on GPU): 30.9826sec