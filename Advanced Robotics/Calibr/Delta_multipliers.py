import numpy as np

class joint_delta():

    def __init__(self, T, dh, joint_ind):
        self.dh = dh
        self.joint_ind = joint_ind
        k = self.k_vectors()
        self.k1n, self.k2n, self.k3n = k

        #Basis rotation and translation vectors:
        self.n = T[0:3, 0]
        self.o = T[0:3, 1]
        self.a = T[0:3, 2]
        self.p = T[0:3, 3]

        #translation : dx, dy, dz:
        self.dx = self.delta_translation(self.n)
        self.dy = self.delta_translation(self.o)
        self.dz = self.delta_translation(self.a)

        #rotation : dx1, dy1, dz1:
        self.dx1 = self.delta_rotation(self.n)
        self.dy1 = self.delta_rotation(self.o)
        self.dz1 = self.delta_rotation(self.a)


    def k_vectors(self):
        "k vectors"
        alpha = self.dh['alpha'][self.joint_ind]
        r = self.dh['d'][self.joint_ind]
        l = self.dh['a'][self.joint_ind]
        k1n = np.array([0, l * np.cos(alpha), -l * np.sin(alpha)]).T
        k2n = np.array([0, np.sin(alpha), np.cos(alpha)]).T
        k3n = np.array([1, 0, 0]).T

        return [k1n, k2n, k3n]


    def delta_translation(self, x):
        """multiplier of translation"""
        x1 = np.dot(x, self.k1n) + np.dot(np.cross(self.p, x), self.k2n)
        x2 = np.dot(x, self.k2n)
        x3 = np.dot(x, self.k3n)
        x4 = np.dot(np.cross(self.p, x), self.k3n)

        return [x1, x2, x3, x4]


    def delta_rotation(self, x):
        """multiplier of rotation"""
        x1 = np.dot(x, self.k2n)
        x2 = np.dot(x, self.k3n)

        return [x1, x2]


def Jacobian(T, dh):
    m1 = np.zeros((3, 6), dtype=np.float32)
    m2 = np.zeros((3, 6), dtype=np.float32)
    m3 = np.zeros((3, 6), dtype=np.float32)
    m4 = np.zeros((3, 6), dtype=np.float32)

    for i in range(6):
        joint_multipliers = joint_delta(T[i], dh, i)

        m = [m1, m2, m3, m4]
        for j in range(4):
            m[j][0, i] = joint_multipliers.dx[j]
            m[j][1, i] = joint_multipliers.dy[j]
            m[j][2, i] = joint_multipliers.dz[j]

    #Jacobian construction:
    jacobian = np.zeros((6, 24), dtype=np.float32)
    jacobian[0:3, 0:6] = m[0]
    jacobian[3:6, 0:6] = m[1]
    jacobian[0:3, 6:12] = m[1]
    jacobian[0:3, 12:18] = m[2]
    jacobian[0:3, 18:24] = m[3]
    jacobian[3:6, 18:24] = m[2]
    return jacobian

