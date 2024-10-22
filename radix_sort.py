# Program: --- python classes (algoriths and data strctures) # 04 ---
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
# RADIX SORT
#


def RadixSort(alist, base):
    def key_factory(digit, base):
        def key(alist, index):
            return (alist[index] // (base ** digit)) % base
        return key
    is_zeros = True
    for i in range(len(alist)):
        is_zeros = is_zeros and alist[i] == 0
    if is_zeros:
        return
    largest = max(alist)
    exp = 0
    while base ** exp <= largest:
        c_result = counting_sort(alist, base - 1, key_factory(exp, base))
        exp = exp + 1
    return


def counting_sort(c_alist, largest, key):
    c = np.zeros(largest + 1, dtype=int)
    for i in range(len(c_alist)):
        c[key(c_alist, i)] = c[key(c_alist, i)] + 1
    c[0] = c[0] - 1
    for i in range(1, largest + 1):
        c[i] = c[i] + c[i - 1]
    result = np.zeros(len(c_alist), dtype=int)
    for i in range(len(c_alist) - 1, -1, -1):
        result[c[key(c_alist, i)]] = c_alist[i]
        c[key(c_alist, i)] = c[key(c_alist, i)] - 1
    for i in range(len(c_alist)):
        c_alist[i] = result[i]  # was c[i]
    return result


def RadixSortCall(array1, call_base):
    time_start = time.time()
    array2 = RadixSort(array1, call_base)
    time_end = time.time()
    return time_end - time_start


# Set parameters
n_min =     10000
n_max =    110000
n_step =    10000
n_repl =      10
n_codes =    500
main_base =  10

n = range(n_min, n_max, n_step)
print(" The length of the range from ", n_min, " to ", n_max, " by ", n_step, " is ", len(n))
t = np.zeros(([len(n), n_repl]), dtype=float)
t2 = np.zeros(([len(n), n_repl]), dtype=float)
t_mean = np.zeros(len(n), dtype=float)
t_SD = np.zeros(len(n), dtype=float)
print(" n = ", n)
print(" t = ", t)


# perform sorting several times in loop
for nn in range(len(n)):
    for i_repl in range(n_repl):
        AA = np.zeros(n[nn], dtype = int)
        for i in range(n[nn]):
            AA[i] = np.floor(n_codes * np.random.rand())
        # print (" Calling sort(AA,1,", n[nn],")")
        # print ( "AA before sort ",AA)
        main_base = 10
        s_time = RadixSortCall(AA, main_base)
        # print ("AA after sort",AA)
        t[nn, i_repl] = s_time
        t2[nn, i_repl] = s_time * s_time
        t_sorted = True
        for i in range(1, n[nn]):
            t_sorted = t_sorted and AA[i] >= AA[i - 1]
        print("N", n[nn],", repl.",i_repl,", sorted" if t_sorted else "unsorted", ", time", t[nn,i_repl])
        if (n[nn] > n_max - n_step) and (i_repl == n_repl - 1):
            print(" Last iteration ")
            nx = np.zeros(n[nn])
            ny = np.zeros(n[nn])
            for i in range(n[nn]):
                nx[i] = i
            t_sorted = True
            ny[0] = True
            for i in range(1, n[nn]):
                t_sorted = t_sorted and AA[i] >= AA[i - 1]
                ny[i] = n_codes * (AA[i] >= AA[i - 1])
            print("N", n[nn],", repl.",i_repl, "sorted" if t_sorted else "unsorted", ", time", t[nn,i_repl])
            plt.plot(nx, AA, 'bo')
            plt.plot(nx, ny, 'r+')
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
# print(t_mean)
# print(t_SD)


plt.plot(n, t, 'bo')
plt.plot(n, t_mean, 'ro')
plt.plot(n, t_mean - t_SD, 'g--')
plt.plot(n, t_mean + t_SD, 'g--')
plt.xlim(0, n_max)
plt.ylim(-0.1, 3.0)
plt.xlabel('n')
plt.ylabel('T(n)')
plt.title('radix-sort')
plt.show()
