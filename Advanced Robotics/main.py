# import roboticstoolbox as rtb
# import numpy as np
# import matplotlib.pyplot as plt
#
# L1 = rtb.RevoluteDH(d=202, a=0, offset=0, alpha=np.pi/2, m=0, r=[0, 0, 0])
# L2 = rtb.RevoluteDH(d=0, a=160, offset=0, alpha=0, m=1, r=[0, 0, 101])
# L3 = rtb.RevoluteDH(d=0, a=0, offset=np.pi/2, alpha=np.pi/2, m=1, r=[0, 0, 80])
# L4 = rtb.RevoluteDH(d=195, a=0, offset=0, alpha=-np.pi/2, m=1, r=[0, 0, 40])
# L5 = rtb.RevoluteDH(d=0, a=0, offset=0, alpha=np.pi/2, m=1, r=[0, 0, 140])
# L6 = rtb.RevoluteDH(d=67.15, a=0, offset=0, alpha=0, m=1, r=[0, 0, 35])
#
# math_robot = rtb.DHRobot([L1, L2, L3, L4, L5, L6])
#
# L1_real = rtb.RevoluteDH(d=205, a=0, offset=0, alpha=np.pi/2, m=0, r=[0, 0, 0])
# L2_real = rtb.RevoluteDH(d=0, a=158, offset=0, alpha=0, m=1, r=[0, 0, 101])
# L3_real = rtb.RevoluteDH(d=0, a=0, offset=np.pi/2, alpha=np.pi/2, m=1, r=[0, 0, 80])
# L4_real = rtb.RevoluteDH(d=194, a=0, offset=0, alpha=-np.pi/2, m=1, r=[0, 0, 40])
# L5_real = rtb.RevoluteDH(d=0, a=0, offset=0, alpha=np.pi/2, m=1, r=[0, 0, 140])
# L6_real = rtb.RevoluteDH(d=69, a=0, offset=0, alpha=0, m=1, r=[0, 0, 35])
#
# real_robot=rtb.DHRobot([L1_real,L2_real,L3_real,L4_real,L5_real,L6_real])
#
# q_zeros = [0, 0, 0, 0, 0, 0]
# q0 = [np.pi/3, np.pi/6,0,np.pi/6,np.pi/6,np.pi/4]
#
# q_used = q0
#
# print(math_robot.fkine(q0))
# print(real_robot.fkine(q0))
#
# print(math_robot.fkine(q_zeros))
# print(real_robot.fkine(q_zeros))
#
# real_robot.plot(q_used,backend='pyplot')
# input("Press Enter to close the plot window...")
import matplotlib.pyplot as plt
import random
import numpy as np
import checking


def data_for_cylinder_along_z(center_x, center_y, radius, height_z):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + center_x
    y_grid = radius * np.sin(theta_grid) + center_y
    return x_grid, y_grid, z_grid


def make_random_step(step_length):
    direction = np.array([random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)])
    vector = direction / np.linalg.norm(direction) * step_length
    return vector


obstacle1 = np.array([0.3, 0.3, 0.1, 0.2])
obstacle_list = [obstacle1]


def check_point_ok(new_coords):
    if (new_coords[0] < -2 or new_coords[1] < -2 or new_coords[2] < -2 or new_coords[0] > 2 or new_coords[1] > 2 or
            new_coords[2] > 2):
        return False
    # dist = np.sqrt((new_coords[0]-obstacle1[0])**2+(new_coords[1]-obstacle1[1])**2)
    # if dist <= obstacle1[2]*1.2:
    #     return False
    colission = checking.check_colission(new_coords, obstacle_list)
    # print(colission)
    return colission


class point:
    def __init__(self, coordinates, parent=None, level=0, children=[]):
        self.coordinates = coordinates
        self.parent = parent
        self.children = children
        self.level = level
        # print(coordinates)

    def create_children(self, number):
        random_step_len = 0.03
        for i in range(number):
            new_coords = self.coordinates + make_random_step(random_step_len)
            while not check_point_ok(new_coords):
                new_coords = self.coordinates + make_random_step(random_step_len)
            new_child = point(new_coords, parent=self, level=self.level + 1,
                              children=[])
            self.children.append(new_child)
            # print(new_child)
        return self.children.copy()


def check_points_to_finish(points, finish_coords):
    for p in points:
        vct = finish_coords - p.coordinates
        dst = np.sqrt(vct[0] ** 2 + vct[1] ** 2 + vct[2] ** 2)

        # print(p.coordinates)
        if dst < 0.03:
            return p
    return None


start_coords = np.array([0., -0.15, 0.1])
finish_coords = np.array([0.5, 0.7, 0.1])

start_point = point(start_coords)

all_points = [start_point]
finish_point = None
while finish_point is None:
    if all_points[0].level % 4 == 0:
        all_points.sort(
            key=lambda x: (x.coordinates[0] - finish_coords[0]) ** 2 + (x.coordinates[1] - finish_coords[1]) ** 2 + (
                        x.coordinates[2] - finish_coords[2]) ** 2)
        # all_points = all_points[:len(all_points) // 2]
        all_points = all_points[:2]

    # for i in range(4):
    new_points = []
    for p in all_points:
        new_points += p.create_children(3)
    finish_point = check_points_to_finish(new_points, finish_coords)
    all_points = new_points

path_points = []

while finish_point is not None:
    path_points.append(finish_point.coordinates)
    finish_point = finish_point.parent

path_points = np.array(path_points[::-1])

# x,y,z = path_points[:,0], path_points[:,1], path_points[:,2]
print(path_points)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
Xc, Yc, Zc = data_for_cylinder_along_z(*obstacle1)
ax.plot_surface(Xc, Yc, Zc, alpha=0.5)
for i in range(len(path_points) - 1):
    ax.plot([path_points[i][0], path_points[i + 1][0]], [path_points[i][1], path_points[i + 1][1]],
            [path_points[i][2], path_points[i + 1][2]])

ax.scatter(path_points[:, 0], path_points[:, 1], path_points[:, 2])
plt.show()

