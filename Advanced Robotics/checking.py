import numpy as np
import matplotlib.pyplot as plt
from spatialmath import SE3
import roboticstoolbox as rtb


def robot_model():
    L1 = rtb.RevoluteDH(d=0.202, a=0, offset=0, alpha=np.pi / 2, m=0, r=[0, 0, 0])
    L2 = rtb.RevoluteDH(d=0, a=0.160, offset=0, alpha=0, m=1, r=[0, 0, 0.101])
    L3 = rtb.RevoluteDH(d=0, a=0, offset=-np.pi / 2, alpha=-np.pi / 2, m=1, r=[0, 0, 0.080])
    L4 = rtb.RevoluteDH(d=0.195, a=0, offset=0, alpha=np.pi / 2, m=1, r=[0, 0, 0.040])
    L5 = rtb.RevoluteDH(d=0, a=0, offset=0, alpha=-np.pi / 2, m=1, r=[0, 0, 0.140])
    L6 = rtb.RevoluteDH(d=0.06715, a=0, offset=0, alpha=0, m=1, r=[0, 0, 0.035])

    math_robot = rtb.DHRobot([L1, L2, L3, L4, L5, L6])
    return math_robot


def robot_points(p):
    "Robot: "
    math_robot = robot_model()
    Nj = len(math_robot.qd)

    "Transformation matrix: "
    T = np.array([[0, 1, 0, p[0]], [0, 0, -1, p[1]], [-1, 0, 0, p[2]], [0, 0, 0, 1]])

    "Inverse kinematics: "
    q_set = math_robot.ik_lm_chan(T)[0]

    "Forward kinematics for all joints: "
    T_list = [np.asarray(math_robot.fkine_all(q_set)[i]) for i in range(Nj)]

    "Robot geometry as a point cloud: "
    N1 = 50  # Number of points in each link
    Robot_points = []
    for j in range(Nj):
        link_points = np.zeros((N1, 3), dtype=np.float32)
        if j != 0:
            for i in range(N1):
                for k in range(3):
                    link_points[i, k] = T_list[j - 1][k, 3] + (T_list[j][k, 3] - T_list[j - 1][k, 3]) * i / N1
        else:
            for i in range(N1):
                for k in range(3):
                    link_points[i, k] = T_list[0][k, 3] * i / N1

        Robot_points.append(link_points)
    return Robot_points, Nj, N1


def check_colission(p, obstacle_list):
    Robot_points, Nj, N1 = robot_points(p)
    for j in range(Nj):
        link_points = Robot_points[j]

        for i in range(N1):
            # some point of robot
            x, y, z = link_points[i, 0], link_points[i, 1], link_points[i, 2]

            # check:
            for obstacle in obstacle_list:
                # if lies in circle:
                if (x - obstacle[0]) ** 2 + (y - obstacle[1]) ** 2 <= obstacle[2] ** 2:
                    # if z coord smaller then height:
                    if z >= 0:
                        if z <= obstacle[3]:
                            return False
                    else:
                        return False

    return True


def plot_robot(p, ax):
    Robot_points, Nj, N1 = robot_points(p)

    for j in range(Nj):
        xmin, xmax = Robot_points[j][0, 0], Robot_points[j][N1 - 1, 0]
        ymin, ymax = Robot_points[j][0, 1], Robot_points[j][N1 - 1, 1]
        zmin, zmax = Robot_points[j][0, 2], Robot_points[j][N1 - 1, 2]

        ax.plot([xmin, xmax], [ymin, ymax], [zmin, zmax], color='m', lw=5)


def plot_obstacles(obstacle_list, ax):
    for obstacle in obstacle_list:
        theta = np.linspace(0, 2 * np.pi, 201)
        r = obstacle[2]
        z = np.linspace(0, obstacle[3], 20)

        thetas, zs = np.meshgrid(theta, z)
        x = r * np.cos(thetas) + obstacle[0]
        y = r * np.sin(thetas) + obstacle[1]
        ax.plot_surface(x, y, zs, color='orange')


if __name__ == "__main__":
    "Obstacles: "  # xc, yc, rc, heigh
    obstacle1 = np.array([0.1, -0.05, 0.2, 0.2])
    obstacle_list = [obstacle1]

    "Coord: "
    p = np.array([0.1, -0.05, 0.1])
    # print(check_colission(p, obstacle_list))
    #
    # math_robot = rtb.models.AL5D()
    # print(math_robot.fkine_all([np.pi / 6 for j in range(4)]))
    fig = plt.figure()
    axes = fig.add_subplot(projection='3d')
    plot_robot(p, axes)
    plot_obstacles(obstacle_list, axes)
    plt.show()
