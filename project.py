
import numpy as np
from ddp import *

n = 100     # System's order
m = 50      # Nmbr of controls

N = 99      # Final time

cost_thres = 0.001

A = np.zeros([n,n])
B = np.zeros([n,m])
C = np.zeros([n,m])

gamma = np.ones(n)

mu = 1.0/200

for i in range(0, n):
    for j in range(0, n):
        if i == j:
            A[i,j] = 0.5
        elif j == i+1:
            A[i,j] = 0.25
        elif j == i-1:
            A[i,j] = -0.25

for i in range(0, n):
    for j in range(0, m):
        B[i,j] = float(i-j)/(n+m)
        C[i,j] = mu * float(i+j)/(n+m)

u_guess = np.ones((N,m))*0.01

x_0 = np.zeros(n)

testSys = testSystem(A, B, C, gamma, n, m)
testCost = testCost(n, m)


# DDP
(x_ddp, u_ddp, J_ddp) = ddp(testSys, testCost, x_0, u_guess, N, cost_thres)

print "Final cost DDP", J_ddp

# Stagewise Newton
(x_new, u_new, J_new) = stagewiseNewton(testSys, testCost, x_0, u_guess, N, cost_thres)

print "Final cost Newton", J_new

