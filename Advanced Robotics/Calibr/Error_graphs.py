import numpy as np
import pandas as pd

import Forward_kinematics
import Function_container
import Delta_multipliers

N = 100
q_points = [np.array([np.random.rand()*np.pi/2 for j in range(6)]) for i in range(N)]
DH_ref = pd.DataFrame(\
    np.array([[0.08946, 0, np.pi/2],
             [0, -0.425, 0],
             [0, -0.3922, 0],
             [0.1091, 0, np.pi/2],
             [0.09465, 0, -np.pi/2],
             [0.0823, 0, 0]]
             ),
    columns = ["d", "a", "alpha"],
    index = ["q1", "q2", "q3", "q4", "q5", "q6"]
)
DH_act = pd.DataFrame(\
    np.array([[0.09, 0, np.pi/2 + np.pi/20],
             [0, -0.43, 0],
             [0, -0.4, 0],
             [0.1, 0, np.pi/2 - np.pi/20],
             [0.1, 0, -np.pi/2 + np.pi/20],
             [0.1, 0, 0]]
             ),
    columns = ["d", "a", "alpha"],
    index = ["q1", "q2", "q3", "q4", "q5", "q6"]
)

#Actual errors list to be filled:
x_error_act = []
y_error_act = []
z_error_act = []
#Modified errors list to be filled:
x_error_mod = []
y_error_mod = []
z_error_mod = []
#Average DH based on N calibration points:
DH_mod_list = []
q_mod_list = []
for q0 in q_points:
    #Step 2: Forward kinematics:
    T_ref = Forward_kinematics.h_transf(DH_ref, q0)
    T_act = Forward_kinematics.h_transf(DH_act, q0)

    #Step 3: Error of initial model:
    dN_act, dN1_act = Function_container.error_calc(T_ref, T_act)
    x_error_act.append(np.abs(dN_act[0]))
    y_error_act.append(np.abs(dN_act[1]))
    z_error_act.append(np.abs(dN_act[2]))
    # Result vector:
    delta = np.vstack((dN_act, dN1_act)).reshape(6, 1)

    #Step 4: Jacobian:
    J = Delta_multipliers.Jacobian(T_ref, DH_ref)

    #Step 5: Solving matrix equation:
    X = np.linalg.pinv(J) @ delta
    q_mod = [q0[i] + X[i, 0] for i in range(6)]
    DH_mod = DH_act.copy()
    DH_mod['d'] -= X[6:12, 0]
    DH_mod['a'] -= X[12:18, 0]
    DH_mod['alpha'] -= X[18:24, 0]

    #Fill calibration list:
    DH_mod_list.append(DH_mod)
    q_mod_list.append(q_mod)

#Step 6:Calculation of forward kinematics and the errors with estimated DH parameters:
DH_mod = DH_mod_list[0]
for i in range(1, N):
    DH_mod = DH_mod.add(DH_mod_list[i], fill_value=0)
DH_mod = DH_mod / N

for q0 in q_points:
    T_ref = Forward_kinematics.h_transf(DH_ref, q0)
    T_mod = Forward_kinematics.h_transf(DH_mod, q0)
    dN_mod, dN1_mod = Function_container.error_calc(T_ref, T_mod)
    x_error_mod.append(np.abs(dN_mod[0]))
    y_error_mod.append(np.abs(dN_mod[1]))
    z_error_mod.append(np.abs(dN_mod[2]))

#Plot Error graphs:
Function_container.plotter(x_error_act, x_error_mod, "The error along X-axis [m]", N=N)
Function_container.plotter(y_error_act, y_error_mod, "The error along Y-axis [m]", N=N)
Function_container.plotter(z_error_act, z_error_mod, "The error along Z-axis [m]", N=N)

#Maximal error:
max_act = Function_container.max_error(x_error_act, y_error_act, z_error_act)
max_mod = Function_container.max_error(x_error_mod, y_error_mod, z_error_mod)

#Average error
ave_act = Function_container.ave_error(x_error_act, y_error_act, z_error_act)
ave_mod = Function_container.ave_error(x_error_mod, y_error_mod, z_error_mod)


#Errors Dataframe:
DH_error = pd.DataFrame(\
    np.array([
        [max_act, max_mod],
        [ave_act, ave_mod]
        ]
             ),
    columns=["Before calibration [m]", "After calibration [m]"],
    index=["maximum change", "average error"]
)
print("\n",DH_error)
print("\n Improvement = ",round((ave_act-ave_mod)/ave_act*100,0),"%")
