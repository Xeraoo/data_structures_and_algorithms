# Program: --- python classes (algorithms and data structures) # 06 ---
# Program sorts random numbers and calculates sorting times

# IMPORT
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

# vector A is of length n - indices start from 0
# First: define heap?ify procedure

def Parent(i):
    return int(np.floor(i / 2.0))

def Left(i):
    return 2 * i

def Right(i):
    return 2 * i + 1

def Heapify(A, i, heap_size):
    l = Left(i); r = Right(i)
    if (l <= heap_size) and (A[l - 1] > A[i - 1]):
        largest = l
    else:
        largest = i
    if (r <= heap_size) and (A[r - 1] > A[largest - 1]):
        largest = r
    if largest != i:
        A[i - 1], A[largest - 1] =  A[largest - 1], A[i - 1]
        Heapify(A, largest, heap_size)

def BuildHeap(A, heap_size):
    for i in range(int(np.floor(heap_size / 2.0)), 0, -1):
        Heapify(A, i, heap_size)

def HeapSort(A, heap_size):
    BuildHeap(A, heap_size)
    for i in range (heap_size, 1, -1):
        A[0], A[i - 1] = A[i - 1], A[0]
        heap_size -= 1
        Heapify(A, 1, heap_size)


def HeapSortCall(A, n):
    time_start = time.time()
    HeapSort(A, n)
    time_end = time.time()
    return time_end - time_start


# Set parameters
date_start = datetime.now()
n_min  =    1000
n_max  =   11000
n_step =    1000
n_repl =      10


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
        AA = np.zeros(n[nn], dtype=float)
        for i in range(n[nn]):
            AA[i] = np.random.rand()
            # AA[i] = 1.0 - float(i) / n[nn]
        # print(" Calling sort(AA, 1, ", n[nn], ")")
        # print(" AA before sort ", AA)
        s_time = HeapSortCall(AA, n[nn])
        # print(" AA after sort ", AA)
        # s_time = s_time ** 0.5
        t[nn, i_repl] = s_time
        t2[nn, i_repl] = s_time * s_time
        # print(" N ", n[nn], " replicate ", i_repl, " time ", t[nn, i_repl])
        t_sorted = True
        for i in range(1, n[nn]):
            t_sorted = t_sorted and AA[i] >= AA[i - 1]
        print("N", n[nn], "NlogN", n[nn] * np.log10(n[nn]), "repl.", i_repl,
              ", sorted." if t_sorted else ", not sorted.", "time", s_time)


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
plt.ylim(-0.05, 1.35)
plt.xlabel('n')
plt.ylabel('T(n)')
plt.title('Sorting time')
date_end = datetime.now()
print(" Calculations started at **>>", date_start, "<<**, ended at **>>", date_end, ",<**")
plt.show()
