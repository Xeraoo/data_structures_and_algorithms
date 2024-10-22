# Program: --- python classes (algorithms and data structures) # 02 ---
# Program sorts random numbers and calculates sorting times


# IMPORT
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import time
from datetime import datetime


# Define sorting algorithm
def QuickSort(A, p, r):
    if p < r:
        q = Partition(A, p, r)
        QuickSort(A, p, q)
        QuickSort(A, q + 1, r)

def Partition(A, p, r):
    x = A[p - 1]  # actually A[p]
    i = p - 1
    j = r + 1
    while True:
        while True:
            i += 1
            if A[i - 1] >= x:
                break  # while A[i] < x
        while True:
            j -= 1
            if A[j - 1] <= x:
                break  # while A[j] > x
        if i < j:
            A[i - 1], A[j - 1] = A[j - 1], A[i - 1]  # swap A[i], A[j]
        else:
            return j


def QuickSortCall(A, left, right):
    time_start = time.time()
    QuickSort(A, left, right)
    time_end = time.time()
    return time_end - time_start


# Set parameters
date_start = datetime.now()
n_min  =    10000
n_max  =   110000
n_step =    10000
n_repl =      10


n  = range(n_min, n_max, n_step)
nlogn = [i for i in range(n_min, n_max, n_step)]
for i in range(len(n)):
    nlogn[i] = nlogn[i] * np.log2(nlogn[i])
print(" The length of the range from ", n_min, " to ", n_max, " by ", n_step, " is ", len(n))
t  = np.zeros(([len(n), n_repl]), dtype=float)
t2 = np.zeros(([len(n), n_repl]), dtype=float)
t_mean = np.zeros(len(n), dtype=float)
t_SD   = np.zeros(len(n), dtype=float)
print(" n = ", n)
print(" nlogn = ", nlogn)


# perform sorting several times in loop
for nn in range(len(n)):
    for i_repl in range(n_repl):
        AA = np.zeros(n[nn], dtype = float)
        for i in range(n[nn]):
            AA[i] = np.random.rand()
        s_time = QuickSortCall(AA, 1, n[nn])
        t_sorted = True
        for i in range(1, n[nn]):
            t_sorted = t_sorted and AA[i] >= AA[i - 1]
        print("N", n[nn], "NlogN", n[nn] * np.log2(n[nn]), "repl.", i_repl,
              ", sorted." if t_sorted else ", not sorted.", "time", s_time)
        t[nn, i_repl] = s_time
        t2[nn, i_repl] = s_time*s_time
        # print(" N " , n[nn] , " replicate ", i_repl, " time " , t[nn, i_repl])
    if n[nn] > 95000:
        print("dddddddddddddddddd", n[nn])
        nx = np.zeros(n[nn])
        for i in range(n[nn]):
            nx[i] = i
        plt.plot(nx, AA, 'bo')
        plt.xlabel('n')
        plt.ylabel('A(n)')
        plt.title('Sorted table')
        plt.show()
    t_mean[nn] = sum(t[nn]) / n_repl
    t_SD[nn] = ((sum(t2[nn]) - sum(t[nn]) * sum(t[nn]) / n_repl) / (n_repl - 1)) ** 0.5
date_end = datetime.now()
print(" Calculations started at **>>", date_start, "<<**, ended at **>>", date_end, ",<**")


print(t_mean)
print(t_SD)
plt.plot(nlogn, t, 'bo')
plt.plot(nlogn, t_mean, 'r')
plt.plot(nlogn, t_mean - t_SD, 'g--')
plt.plot(nlogn, t_mean + t_SD, 'g--')
plt.xlim(0, n_max * np.log2(n_max))
plt.ylim(-0.1, 2.0)
plt.xlabel('n')
plt.ylabel('T(n)')
plt.title('Sorting time')
date_end = datetime.now()
print(" Calculations started at **>>", date_start, "<<**, ended at **>>", date_end, ",<**")
plt.show()

