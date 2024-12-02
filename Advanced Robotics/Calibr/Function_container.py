import numpy as np
import matplotlib.pyplot as plt


def angle(vector_1,vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.arccos(dot_product)


def plotter(act, mod, title, N=100):
    plt.rcParams['font.size'] = 24# Option 2
    plt.rcParams.update({'font.size': 24})
    plt.figure(figsize=(10,6))
    plt.plot([i for i in range(1,N+1)], act, label = "Initial", color = 'r')
    plt.plot([i for i in range(1,N+1)], mod, label = "Modified", color = 'b')
    plt.legend()
    plt.title(title)
    plt.show()


def error_calc(T_ref, T):
    # Translation error:
    dN = np.array([ \
        -T_ref[5][0, 3] + T[5][0, 3],
        -T_ref[5][1, 3] + T[5][1, 3],
        -T_ref[5][2, 3] + T[5][2, 3]
    ]).T
    # Rotation error:
    dN1 = np.array([ \
        angle(T[5][0:3, 0], T_ref[5][0:3, 0]),
        angle(T[5][0:3, 1], T_ref[5][0:3, 1]),
        angle(T[5][0:3, 2], T_ref[5][0:3, 2])
    ]).T
    return dN, dN1


def max_error(x, y, z):
    array = np.array([x, y, z])
    return array.max()


def ave_error(x, y, z):
    array = np.array([x, y, z])
    return np.average(array)
