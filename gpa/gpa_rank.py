import numpy as np
import procrustes as prc

# input score-differential matrix
A = np.array([[0, 0, 0, 0, 0],  # Duke
              [45, 0, 18, 8, 20],  # Miami
              [3, 0, 0, 2, 0],  # UNC
              [31, 0, 0, 0, 0],  # UVA
              [45, 0, 27, 38, 0]])  # VT

# make rank-differential matrix
n = A.shape[0]
B = np.zeros((n, n))
for index in range(n):
    B[index, index:] = range(0, n - index)

# rank teams using two-sided Procrustes
result = prc.ValidationError.(A, B, single=True, method="approx-normal1")

# compute teams' ranks
_, ranks = np.where(result.t == 1)
ranks += 1
print("Ranks = ", ranks)  # displays [5, 2, 4, 3, 1]
