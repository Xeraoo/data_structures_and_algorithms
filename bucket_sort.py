# Program: --- python classes (algorithms and data structures) # 09 ---
# Program sorts random numbers and calculates sorting times


# IMPORT
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import time
from datetime import datetime
from scipy import special


# Define sorting algorithm
# Sort n numbers in ascending order and calculate time
# vector A is of length n - indices start from 0

# INSERTION SORT - auxiliary
def InsertionSort(A_IS, n_IS):
    for i in range(1, n_IS):
        key = A_IS[i]
        j = i - 1
        while ((j >= 0) and (A_IS[j] > key)):
            A_IS[j + 1] = A_IS[j]
            j = j - 1
        A_IS[j + 1] = key

#
# BUCKET SORT
#

def BucketSort(A, n, N_Limits, distr):
    # print("Limits separate consecutive ", N_Limits, " intervals.")
    n_of_intervals = N_Limits                        #  no. of "buckets"
    x_limit = np.zeros(N_Limits + 1, dtype=np.float) #  x-s for limits
    x_limit = [i for i in range(len(x_limit))]
    Limit = np.zeros(N_Limits + 1, dtype=np.float)   #  lower limits of bucket
    Bucket_limit = np.zeros(N_Limits, dtype=np.int)  #  Max. bucket size
    Bucket_size  = np.zeros(N_Limits, dtype=np.int)  #  Actual bucket size
    for bucket_no in range(n_of_intervals):
        Bucket_limit[bucket_no] = buck_overf * n / n_of_intervals
    # print(" Bucket size limits : ", Bucket_limit)
    Bucket = np.zeros((n_of_intervals, Bucket_limit[0]), dtype = np.float)
    if distr == "UNI": # uniform [0; 1]
        for i in range(n_of_intervals):
            Limit[i] = max(A) * float(i) / float(n_of_intervals)
        y = Limit
    if distr == "NOR": # normal (0, 1)
        for i in range(n_of_intervals + 1):
            Limit[i] = 2.0 * float(i) / float(n_of_intervals) - 1.0
        y = special.erfinv(Limit)
    Limit[n_of_intervals] = np.inf
    # Fill buckets
    for i in range(n):
        bucket_number = 0
        while A[i] >= Limit[bucket_number]:
            bucket_number += 1
        bucket_number -= 1
        Bucket[bucket_number, Bucket_size[bucket_number]] = A[i]
        Bucket_size[bucket_number] += 1
    # print(" Sizes of buckets: ", Bucket_size)
    elem_no = 0
    for i in range(N_Limits):
        InsertionSort(Bucket[i], Bucket_size[i])
        for j in range(Bucket_size[i]):
            A[elem_no] = Bucket[i][j]
            elem_no += 1
    # print("No of elems = ", elem_no)


def BucketSortCall(A, n, n_lim, distr_type):
    time_start = time.time()
    BucketSort(A, n, n_lim, distr_type)
    time_end = time.time()
    return time_end - time_start

# Set parameters
date_start = datetime.now()
n_min      =   500
n_max      =  5500
n_step     =   500
n_repl     =    10
n_buckets  =   100
buck_overf =    10  #  eg. "2" means bucket holds up to 2x expected n. of elements
dist_type  =  "UNI"

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
        s_time = BucketSortCall(AA, n[nn], n_buckets, dist_type)
        t[nn][i_repl]  = s_time
        t2[nn][i_repl] = s_time * s_time
        t_sorted = True
        for i in range(1, n[nn]):
            t_sorted = t_sorted and AA[i] >= AA[i - 1]
        print(n[nn], t_sorted, s_time)
    if n[nn] == n_max - n_step:
        nx = np.zeros(n[nn], dtype=np.float32)
        ny = np.zeros(n[nn], dtype=np.float32)
        for i in range(n[nn]):
            nx[i] = i
        ny[0] = float(n[nn])
        for i in range(1, n[nn]):
            ny[i] = float(n[nn]) * float(AA[i] >= AA[i - 1])
        plt.plot(nx, n[nn] * AA, 'bo')
        plt.plot(nx, ny, 'r.', )
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
plt.plot(n, t, 'bo')
plt.plot(n, t_mean, 'r')
plt.plot(n, t_mean - t_SD, 'g--')
plt.plot(n, t_mean + t_SD, 'g--')
plt.xlim(0, n_max)
plt.ylim(-0.05, 0.65)
plt.xlabel('n')
plt.ylabel('T(n)')
plt.title('Sorting time')
date_end = datetime.now()
print(" Calculations started at **>>", date_start, "<<**, ended at **>>", date_end, ",<**")
plt.show()
