

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
# Graph searching - BFS algorithm
#


def Q_empty(Q):
    return len(Q) == 0


def enqueue(Q, v):
    Q.append(v)


def dequeue(Q):
    if Q == []:
        return []
    else:
        return Q.pop(0)


def BFS(A, v, n, n_0):
    time_start = time.time()
    v_color = ["white" for i in range(n)]
    Q = [n_0]
    while not(Q_empty(Q)):
        u = dequeue(Q)
        print( " The dequeued element is ", u, " and the queue is ", Q)
        for v in range(n):
            if (v != u) and (A[u, v] > 0):
                if v_color[v] == "white":
                    v_color[v] = "grey"
                    v_d[v] = v_d[u] + 1
                    v_p[v] = u
                    enqueue(Q, v)
                    print(" The enqueued element is ", v, " and the queue is ", Q)
        v_color[v] = "black"
    time_end = time.time()
    return time_end - time_start


def GraphDraw(A, x, y, v, n):
    # Draw graph with vertices x,y,v and edges A
    # VERTICES
    plt.plot(x, y, 'yo', ms=30)
    # EDGES
    for i in range(n - 1):
        for j in range(i + 1, n):
            # DRAW A LINE o-o-o (i,j)
            if A[i, j] > 0:
                plt.plot((x[i],x[j]),(y[i],y[j]), color='black')
    plt.xlim(-0.2, 1.2)
    plt.ylim(-0.2, 1.2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Drawing graph of ' + str(n) + ' vertices')
    ax = plt.subplot()
    for i, txt in enumerate(v):
        ax.annotate(txt, (x[i], y[i]))
    plt.savefig('graph_BFS_' + str(n) + '.png', bbox_inches='tight')
    plt.show()
    plt.close()

    # There is the A matrix [nxn] and x, y, v vectors [n]


def GraphDrawCall(A, x, y, v, n):
    time_start = time.time()
    # GraphDraw(A, x, y, v, n)
    time_end = time.time()
    return time_end - time_start


# Set parameters
date_start = datetime.now()
n_min   =    14
n_max   =    16
n_step  =     1
n_repl  =     5
ThresholdProportion = 0.5
Q = []


n = range(n_min, n_max, n_step)
print(" The length of the range from ", n_min, " to ", n_max, " by ", n_step, " is ", len(n))
t = np.zeros(([len(n), n_repl]), dtype=float)
t2 = np.zeros(([len(n), n_repl]), dtype=float)
t_mean = np.zeros(len(n), dtype=float)
t_SD = np.zeros(len(n), dtype=float)
print(" n = ", n)
# perform drawing several times in loop
for nn in range(len(n)):
    for i_repl in range(n_repl):
        # AA - adjacency matrix
        # v_d, v_p - distance matrix and parent matrix
        AA = np.zeros((n[nn], n[nn]), dtype=float)
        v_d = np.zeros(n[nn], dtype=int)
        v_p = np.zeros(n[nn], dtype=int)
        xx = np.zeros(n[nn], dtype=float)
        yy = np.zeros(n[nn], dtype=float)
        vv = np.zeros(n[nn], dtype=int)
        for i in range(n[nn] - 1):
            for j in range(i + 1, n[nn]):
                if np.random.rand() > (1.0 - ThresholdProportion) ** 0.5:
                    AA[i, j] = np.random.rand()
                    AA[j, i] = AA[i, j]
        for i in range(n[nn]):
            xx[i] = np.random.rand()
            yy[i] = np.random.rand()
            vv[i] = i
        # print(AA, xx, yy)
        s_time = GraphDrawCall(AA, xx, yy, vv, n[nn])
        # convert graph into list (set) implementation:
        graph = dict()
        for i in range(n[nn]):
            node_list = []
            for j in range(n[nn]):
                if AA[i, j] > 0.0:
                    node_list.append(j)
            graph.update({str(i): node_list})
        # print(" The input graph is: ", graph)
        s_time2 = BFS(AA, vv, n[nn], int(np.random.rand() * n[nn]))
        t[nn, i_repl] = s_time2
        t2[nn, i_repl] = s_time2 * s_time2
        print("Draw: N", n[nn], "replicate", i_repl, "time", t[nn, i_repl])
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
plt.xlim(n_min - 1, n_max)
plt.ylim(-0.001, 0.005)
plt.xlabel('n')
plt.ylabel('T(n)')
plt.title('Calculating time')
date_end = datetime.now()
print(" Calculations started at **>>", date_start, "<<**, ended at **>>", date_end, ",<**")
plt.show()
