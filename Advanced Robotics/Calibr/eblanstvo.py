import numpy as np


def Rx(q):
    T = np.array([[1, 0, 0, 0],
                  [0, np.cos(q), -np.sin(q), 0],
                  [0, np.sin(q), np.cos(q), 0],
                  [0, 0, 0, 1]])
    return T


def Ry(q):
    T = np.array([[np.cos(q), 0, np.sin(q), 0],
                  [0, 1, 0, 0],
                  [-np.sin(q), 0, np.cos(q), 0],
                  [0, 0, 0, 1]])
    return T


def Rz(q):
    T = np.array([[np.cos(q), -np.sin(q), 0, 0],
                  [np.sin(q), np.cos(q), 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    return T


def Tx(x):
    T = np.array([[1, 0, 0, x],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    return T


def Ty(y):
    T = np.array([[1, 0, 0, 0],
                  [0, 1, 0, y],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    return T


def Tz(z):
    T = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, z],
                  [0, 0, 0, 1]])
    return T


def dRx(q):
    T = np.array([[0, 0, 0, 0],
                  [0, -np.sin(q), -np.cos(q), 0],
                  [0, np.cos(q), -np.sin(q), 0],
                  [0, 0, 0, 0]])
    return T


def Rdy(q):
    T = np.array([[-np.sin(q), 0, np.cos(q), 0],
                  [0, 0, 0, 0],
                  [-np.cos(q), 0, -np.sin(q), 0],
                  [0, 0, 0, 0]])
    return T


def Rdz(q):
    T = np.array([[-np.sin(q), -np.cos(q), 0, 0],
                  [np.cos(q), -np.sin(q), 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])
    return T


def Tdx(x):
    T = np.array([[0, 0, 0, 1],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])
    return T


def Tdy(y):
    T = np.array([[0, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])
    return T


def Tdz(z):
    T = np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0]])
    return T


def Jth(transmission, deflections, flag):
    t01 = transmission['t01']
    t12 = transmission['t12']
    t23 = transmission['t23']

    if flag == 'J1':
        T01 = Rz(t01[0]) @ Rdz(deflections[0]
                 ) @ Tz(t01[1]) @ Tx(t01[2]) @ Rx(t01[3])
        T12 = Rz(t12[0]) @ Tz(t12[1]) @ Tz(deflections[1]
                 ) @ Tx(t12[2]) @ Rx(t12[3])
        T23 = Rz(t23[0]) @ Tz(t23[1]) @ Tz(deflections[2]
                 ) @ Tx(t23[2]) @ Rx(t23[3])

        T03 = T01 @ T12 @ T23

        Jth = np.array([T03[0, 3], T03[1, 3], T03[2, 3],
                       T03[2, 1], T03[0, 2], T03[1, 0]])

    elif flag == 'J2':
        T01 = Rz(t01[0]) @ Rz(deflections[0]
                 ) @ Tz(t01[1]) @ Tx(t01[2]) @ Rx(t01[3])
        T12 = Rz(t12[0]) @ Tz(t12[1]) @ Tdz(deflections[1]
                 ) @ Tx(t12[2]) @ Rx(t12[3])
        T23 = Rz(t23[0]) @ Tz(t23[1]) @ Tz(deflections[2]
                 ) @ Tx(t23[2]) @ Rx(t23[3])

        T03 = T01 @ T12 @ T23

        Jth = np.array([T03[0, 3], T03[1, 3], T03[2, 3],
                       T03[2, 1], T03[0, 2], T03[1, 0]])

    elif flag == 'J3':
        T01 = Rz(t01[0]) @ Rz(deflections[0]
                 ) @ Tz(t01[1]) @ Tx(t01[2]) @ Rx(t01[3])
        T12 = Rz(t12[0]) @ Tz(t12[1]) @ Tz(deflections[1]
                 ) @ Tx(t12[2]) @ Rx(t12[3])
        T23 = Rz(t23[0]) @ Tz(t23[1]) @ Tdz(deflections[2]
                 ) @ Tx(t23[2]) @ Rx(t23[3])

        T03 = T01 @ T12 @ T23

        Jth = np.array([T03[0, 3], T03[1, 3], T03[2, 3],
                       T03[2, 1], T03[0, 2], T03[1, 0]])

    return Jth


def inverse_cyl(pose, transmission):
    x = pose[0]
    y = pose[1]
    z = pose[2]

    a = transmission['a']
    b = transmission['b']

    q1 = np.arctan2(y, x)
    q2 = z - a
    q3 = np.sqrt(x**2 + y**2) - b

    q = np.array([q1, q2, q3])

    return q


# Define parameters
iterations = 10
a = 2
b = 1
ql = np.array([0, 0])
qu = np.array([2*np.pi, 1])
fl = -1000
fu = 1000
deflections = np.zeros(3)
K = 1e6 * np.diag([1, 2, 0.5])

transmission = {'a': a, 'b': b}

# Initialize arrays for storing results
forces = np.zeros((3, iterations))
poses = np.zeros((3, iterations))
DTs = np.zeros((3, iterations))

# Perform iterations
info_sum = 0
AT_sum = 0
for i in range(iterations):
    # Generate random force and random position
    th1 = ql[0] + (qu[0] - ql[0]) * np.random.rand()
    d1d2 = ql[1:] + (qu[1:] - ql[1:]) * np.random.rand(2)
    F_rand = fl + (fu - fl) * np.random.rand(3)

    # Append values to DH parameter
    transmission['t01'] = np.array([th1, 0, 0, 0])
    transmission['t12'] = np.array([-np.pi/2, d1d2[0]+a, 0, -np.pi/2])
    transmission['t23'] = np.array([0, d1d2[1]+b, 0, 0])

    # Assemble wrench vector
    W = np.append(F_rand, np.zeros(3))

    # Append forces and poses in arrays
    forces[:, i] = W[:3]
    poses[:, i] = np.array([th1, d1d2[0], d1d2[1]])

    # Calculate Jacobians
    Jth1 = Jth(transmission, deflections, 'J1')
    Jth2 = Jth(transmission, deflections, 'J2')
    Jth3 = Jth(transmission, deflections, 'J3')

    Jth_t = np.vstack((Jth1, Jth2, Jth3))

    # Calculate delta_t
    dt = np.dot(np.dot(np.dot(Jth_t.T, np.linalg.inv(K)), Jth_t), W) + np.random.normal(0, 1e-5)
    # print(dt.shape)
    # Append delta_t in array
    #DTs[:, i] = dt

    # Calculate matrix A
    a1 = np.dot(np.dot(Jth1, Jth1.T),(W))
    a2 = np.dot(np.dot(Jth2, Jth2.T),(W))
    a3 = np.dot(np.dot(Jth3, Jth3.T),(W))
    A = np.vstack((a1, a2, a3))
    # print(A.shape)
    info_mat = A.dot(A.T)
    AT = A.dot(dt)
    info_sum += info_mat
    AT_sum += AT
    # print(info_mat.shape)
# Calculate estimated stiffness
# print(info_sum.shape)
k_hat = np.linalg.inv(info_sum).dot(AT_sum)

stiffness = np.diag(1/k_hat)

# Define circular path
t = np.linspace(0, 2*np.pi)
x = np.cos(t)
y = np.sin(t)
z = np.zeros_like(x) + 2.5
pnts = np.vstack((x, y, z))

# Assemble wrench vector
W = np.array([500, 500, -500, 0, 0, 0])

uncalib_poses = np.zeros_like(pnts)
DTs_uncal = np.zeros((6, len(t)))

# Simulate robot along path with calibration
for i in range(len(t)):
    pose = np.array([x[i], y[i], z[i]])
    qs = inverse_cyl(pose, transmission)

    transmission['t01'] = np.array([qs[0], 0, 0, 0])
    transmission['t12'] = np.array([-np.pi/2, qs[1]+a, 0, -np.pi/2])
    transmission['t23'] = np.array([0, qs[2]+b, 0, 0])

    # Calculate Jacobians
    Jth1 = Jth(transmission, deflections, 'J1')
    Jth2 = Jth(transmission, deflections, 'J2')
    Jth3 = Jth(transmission, deflections, 'J3')

    Jth_t = np.vstack((Jth1, Jth2, Jth3))
    # Calculate delta_t
    print(Jth_t.shape, np.linalg.inv(stiffness), W.shape)
    #eq1=np.dot(Jth_t.T, np.linalg.inv(stiffness))
    #eq2=np.dot(eq1,Jth_t)
    #eq3=np.dot(eq2,W)

    #dt_uncalib = eq3 + np.random.normal(0, 1e-5)
    dt_uncalib = np.dot(np.dot(np.dot(Jth_t.T, np.linalg.inv(stiffness)), Jth_t), W) + np.random.normal(0, 1e-5)
    print(dt_uncalib.shape)
    uncalib_pose=pose + dt_uncalib[:3]
    print(DTs_uncal.shape)
    # Append delta_t in array
    DTs_uncal[:, i]=dt_uncalib

    uncalib_poses[:, i]=uncalib_pose

# Compensate for deflection
difference=pnts.T - uncalib_poses.T
calibrated=pnts.T + difference

# Plot results
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig=plt.figure()
ax=fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, '--r', label='Target path', linewidth=2)
ax.scatter(uncalib_poses[0], uncalib_poses[1], uncalib_poses[2],
           c='k', label='Real path')
ax.scatter(calibrated[:, 0], calibrated[:, 1], calibrated[:, 2],
           c='b', label='Compensated path')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
plt.show()
