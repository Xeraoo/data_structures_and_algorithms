# Program: --- python classes (algorithms and data structures) # 09 ---
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


#
# COUNTING SORT
#


def CountingSort(A, n, n_codes):
    # print(" A = ", A)
    # Create table of codes
    t_codes = np.zeros(n_codes + 1, dtype=int)
    # create work table of sorted data
    B = np.zeros(n, dtype=int)
    for i in range(n):
        t_codes[A[i]] += 1
    # print(" t_codes ", t_codes)
    for i in range(n_codes + 1):
        t_codes[i] = t_codes[i] + t_codes[i - 1]
    # print(" t_codes[2] ", t_codes)
    for i in range(n, 0, -1):
        B[t_codes[A[i - 1]] - 1] = A[i - 1]
        t_codes[A[i - 1]] -= 1
    # print(" B = ", B)
    for i in range(n):
        A[i] = B[i]


def CountingSortCall(A, n, n_codes):
    time_start = time.time()
    CountingSort(A, n, n_codes)
    time_end = time.time()
    return time_end - time_start



################ BUBBLE
def BubbleSort(A, n):
    time_start = time.time()
    sorted = False
    while (not sorted):
        sorted = True
        for i in range(n - 1):
            if A[i] > A[i+1]:
                A[i], A[i + 1] = A[i + 1], A[i]
            sorted = False
    time_end = time.time()
    return time_end - time_start
###################### BUBBLE END


# Set parameters
date_start = datetime.now()
n_min   =   1000
n_max   =  41000
n_step  =   1000
n_repl  =     10
n_codes =    500


n = range(n_min, n_max, n_step)
print(" The length of the range from ", n_min, " to ", n_max, " by ", n_step, " is ", len(n))
t = np.zeros(([len(n), n_repl]), dtype=float)
t2 = np.zeros(([len(n), n_repl]), dtype=float)
t_mean = np.zeros(len(n), dtype=float)
t_SD = np.zeros(len(n), dtype=float)
print(" n = ", n)


# perform sorting several times in loop
for nn in range(len(n)):
    for i_repl in range(n_repl):
        AA = np.zeros(n[nn], dtype=int)
        AA1 = np.zeros(n[nn], dtype=int)
        for i in range(n[nn]):
            AA[i] = np.floor(n_codes * np.random.rand())
        s_time = CountingSortCall(AA, n[nn], n_codes)
        t_sorted = True
        for i in range(1, n[nn]):
              t_sorted = t_sorted and AA[i-1] <= AA[i]
        t[nn, i_repl] = s_time
        t2[nn, i_repl] = s_time * s_time
        print(" N ", n[nn], " replicate ", i_repl, " sorted:", t_sorted, ", time ", t[nn, i_repl])
    if n[nn] > 39999:
        nx = np.zeros(n[nn])
        ny = np.zeros(n[nn])
        for i in range(n[nn]):
            nx[i] = i
        t_sorted = True
        ny[0] = n_codes
        for i in range(1, n[nn]):
            t_sorted = t_sorted and (AA[i] >= AA[i - 1])
            ny[i] = n_codes * (AA[i] >= AA[i - 1])
        plt.plot(nx, AA, 'bo')
        plt.plot(nx, ny, 'r.')
        plt.xlabel('n')
        plt.ylabel('A(n)')
        plt.title('Sorted table')
        plt.show()
        if t_sorted:
            print("\n The table is sorted \n ")
        else:
            print(" The table is not sorted ")
    t_mean[nn] = sum(t[nn]) / n_repl
    t_SD[nn] = ((sum(t2[nn]) - sum(t[nn]) * sum(t[nn]) / n_repl) / (n_repl - 1)) ** 0.5
date_end = datetime.now()
print(" Calculations started at **>>", date_start, "<<**, ended at **>>", date_end, ",<**")


print(t_mean)
print(t_SD)
plt.plot(n, t, 'bo')
plt.plot(n, t_mean, 'r')
plt.plot(n, t_mean - t_SD, 'g--')
plt.plot(n, t_mean + t_SD, 'g--')
plt.xlim(0, n_max)
plt.ylim(-0.01, 0.1)
plt.xlabel('n')
plt.ylabel('T(n)')
plt.title('Sorting time')
date_end = datetime.now()
print(" Calculations started at **>>", date_start, "<<**, ended at **>>", date_end, ",<**")
plt.show()
