# Program: --- python classes (algorithms and data structures) # 01 ---
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
def SelectionSort(A, n):
    time_start: float = time.time()
    for i in range(len(A)):
        min_idx = i
        for j in range(i +1, len(A)):
            if A[min_idx] > A[j]:
                min_idx = j
        A[i], A[min_idx] = A[min_idx], A[i]
    time_end = time.time()
    return time_end - time_start


# Define plotting
def plot_sorted(tab_A):
    x_A = np.zeros(len(tab_A), dtype=float)
    for ind_A in range(len(x_A)):
        x_A[ind_A] = ind_A
    plt.plot(x_A, tab_A)
    plt.xlabel('x_A')
    plt.ylabel('tab_A')
    plt.title('Insertion Sort')
    plt.show()


# Set parameters
date_start = datetime.now()
print("Początek programu, czas: ", date_start)

n_min = 100
n_max = 1600
n_step = 100
n_repl = 10

n = range(n_min, n_max, n_step)
print(" The length of the range from ", n_min, " to ", n_max, " by ", n_step, " is ", len(n))

t = np.zeros(([len(n), n_repl]), dtype=float)
t2 = np.zeros(([len(n), n_repl]), dtype=float)
t_mean = np.zeros(len(n), dtype=float)
t_SD = np.zeros(len(n), dtype=float)

# perform sorting several times in loop
for nn in range(len(n)):
    for i_repl in range(n_repl):
        AA = np.zeros(n[nn])
        for i in range(n[nn]):
            AA[i] = np.random.rand()
        s_time = SelectionSort(AA, n[nn])
        t[nn, i_repl] = s_time
        t2[nn, i_repl] = s_time * s_time
        print(" N ", n[nn], " replicate ", i_repl, " time ", t[nn, i_repl])
    t_mean[nn] = sum(t[nn]) / n_repl
    t_SD[nn] = ((sum(t2[nn]) - sum(t[nn]) * sum(t[nn]) / n_repl) / (n_repl - 1)) ** 0.5

plot_sorted(AA)

print(t_mean)
print(t_SD)

plt.plot(n, t, 'b.')
plt.plot(n, t_mean, 'r')
plt.plot(n, t_mean - t_SD, 'g--')
plt.plot(n, t_mean + t_SD, 'g--')
plt.xlim(0, n_max)
plt.ylim(-0.1, 0.8)
plt.xlabel('n')
plt.ylabel('T(n)')
plt.title('Selection Sort')

date_end = datetime.now()
print(" Selection Sort Calculations started at **>>", date_start, "<<**, ended at **>>", date_end, ",<<**")
plt.show()