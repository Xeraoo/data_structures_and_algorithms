# Program: --- python classes (algorithms and data structures) # 02 ---
# Program sorts random numbers and calculates sorting times

# IMPORT
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Define sorting algorithm
# Sort n numbers in ascending order and calculate time
# vector A is of length n - indices start from 0
def Merge(A, left, mid, right):
    id_L = left; id_R = mid + 1; id_B = 0
    B = np.zeros(right - left + 1, dtype = float)
    while (id_L <= mid) and (id_R <= right):
        if A[id_L - 1] <= A[id_R - 1]:
            B[id_B] = A[id_L - 1]; id_L += 1; id_B += 1
        else:
            B[id_B] = A[id_R - 1]; id_R += 1; id_B += 1
    for i in range(id_L, mid + 1):
        B[id_B] = A[i - 1]; id_B += 1
    for i in range(id_R, right + 1):
        B[id_B] = A[i - 1]; id_B += 1
    for i in range(len(B)):
        A[left + i - 1] = B[i]

def MergeSort(A, left, right):
    if left < right:
        mid = int(np.floor((left + right) / 2.0))
        MergeSort(A, left, mid)
        MergeSort(A, mid + 1, right)
        Merge(A, left, mid, right)

def MergeSortCall(A, left, right):
    time_start = time.time()
    MergeSort(A, left, right)
    time_end = time.time()
    return time_end - time_start

# Set parameters
date_start = datetime.now()
n_min  =   1000
n_max  =  15000
n_step =   1000
n_repl =     10

n  = range(n_min, n_max, n_step)
print(" The length of the range from ", n_min, " to ", n_max, " by ", n_step, " is ", len(n))
t  = np.zeros(([len(n), n_repl]), dtype=float)
t2 = np.zeros(([len(n), n_repl]), dtype=float)
t_mean = np.zeros(len(n), dtype=float)
t_SD   = np.zeros(len(n), dtype=float)
print(" n = ", n)

# perform sorting several times in loop
for nn in range(len(n)):
    for i_repl in range(n_repl):
        AA = np.zeros(n[nn])
        for i in range(n[nn]):
            AA[i] = np.random.rand()
        s_time = MergeSortCall(AA, 1, n[nn])
        t[nn, i_repl] = s_time
        t2[nn, i_repl] = s_time*s_time
        t_sorted = True
        for i in range(1, n[nn]):
            t_sorted = t_sorted and AA[i] >= AA[i - 1]
        print("N" , n[nn] , "replicate", i_repl, ", sorted." if t_sorted else ", not sorted.", "time" , t[nn, i_repl])
    t_mean[nn] = sum(t[nn]) / n_repl
    t_SD[nn] = ((sum(t2[nn]) - sum(t[nn]) * sum(t[nn]) / n_repl) / (n_repl - 1)) ** 0.5
date_end = datetime.now()

plt.plot(n, t, 'bo')
plt.plot(n, t_mean, 'r')
plt.plot(n, t_mean - t_SD, 'g--')
plt.plot(n, t_mean + t_SD, 'g--')
plt.xlim(0, n_max)
plt.ylim(-0.05, 0.35)
plt.xlabel('n')
plt.ylabel('T(n)')
plt.title('Sorting time')
date_end = datetime.now()
print(" Calculations started at **>>", date_start, "<<**, ended at **>>", date_end, ",<**")
plt.show()

