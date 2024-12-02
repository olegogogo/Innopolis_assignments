import numpy as np


def forward_kine(dh, q_set):
    """transformation matrices from joint to joint"""
    q1, q2, q3, q4, q5, q6 = q_set

    def joint_to_joint(q, alpha, r, l):
        mat = np.array(\
            [[np.cos(q), -np.sin(q) * np.cos(alpha), np.sin(q) * np.sin(alpha), l * np.cos(q)],
                       [np.sin(q), np.cos(q) * np.cos(alpha), -np.cos(q) * np.sin(alpha), l * np.sin(q)],
                       [0, np.sin(alpha), np.cos(alpha), r],
                       [0, 0, 0, 1]]
                      )
        return mat

    a01 = joint_to_joint(q1, dh['alpha'][0], dh['d'][0], dh['a'][0])
    a12 = joint_to_joint(q2, dh['alpha'][1], dh['d'][1], dh['a'][1])
    a23 = joint_to_joint(q3, dh['alpha'][2], dh['d'][2], dh['a'][2])
    a34 = joint_to_joint(q4, dh['alpha'][3], dh['d'][3], dh['a'][3])
    a45 = joint_to_joint(q5, dh['alpha'][4], dh['d'][4], dh['a'][4])
    a56 = joint_to_joint(q6, dh['alpha'][5], dh['d'][5], dh['a'][5])

    a = [a01, a12, a23, a34, a45, a56]

    return a


def h_transf(DH, q_set):
    """Homogeneous transformation matrices from base to joint"""
    a = forward_kine(DH, q_set)
    t = []
    for i in range(len(a)):
        temp = a[0]
        for j in range(i):
            temp = temp @ a[j+1]
        t.append(temp)
    return t
