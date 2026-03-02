'''
    Elastic equilibrium equation
    Please refer to Section 3.3.
    If you have any question, please don’t hesitate to contact us (23121084@bjtu.edu.cn).
    When more than 200 points are used together with sine functions, the results are accurate.
'''

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
import xlrd
import os
import matplotlib.pylab as plt
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # run the code with the CPU

# tf.set_random_seed(1)


def MLP1(x, reuse=None):# In this test setting, the DEM with ReLu function has the best performance.
    with tf.variable_scope('MLnx', reuse=reuse):
        l1 = tf.layers.dense(x, 64)
        l1 = tf.nn.relu(l1)
        l2 = tf.layers.dense(l1, 64)
        l2 = tf.nn.relu(l2)
        l3 = tf.layers.dense(l2, 64)
        l3 = tf.nn.relu(l3)
        output = tf.layers.dense(l3, 1)
    return output


def MLP2(x, reuse=None):
    with tf.variable_scope('MLny', reuse=reuse):
        l1 = tf.layers.dense(x, 64)
        l1 = tf.nn.relu(l1)
        l2 = tf.layers.dense(l1, 64)
        l2 = tf.nn.relu(l2)
        l3 = tf.layers.dense(l2, 64)
        l3 = tf.nn.relu(l3)
        output = tf.layers.dense(l3, 1)
    return output


def PINN(x, y, reuse=None):

    with tf.GradientTape(persistent=True) as gg:
        gg.watch(x)
        gg.watch(y)
        U = MLP1(tf.concat([x, y], axis=1), reuse=reuse) * x
        V = MLP2(tf.concat([x, y], axis=1), reuse=reuse) * y
    Ux = gg.gradient(U, x)
    Vy = gg.gradient(V, y)
    Uy = gg.gradient(U, y)
    Vx = gg.gradient(V, x)
    Uxy = Uy + Vx
    del gg

    txx = E / (1 - v * v) * (Ux + v * Vy)
    tyy = E / (1 - v * v) * (Vy + v * Ux)
    txy = E / 2 / (1 + v) * (Uxy)

    return U, V, txx, tyy, txy, Ux, Vy, Uxy, Uy, Vx


def generate_boundary_points(R, r, n_points_per_edge):

    boundary_points = []
    boundary_types = []

    angles = np.linspace(0, np.pi / 2, n_points_per_edge, endpoint=True)
    for theta in angles:
        x = R * np.cos(theta)
        y = R * np.sin(theta)
        boundary_points.append([x, y])
        boundary_types.append('outer')

    angles = np.linspace(0, np.pi / 2, n_points_per_edge, endpoint=True)
    for theta in angles:
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        boundary_points.append([x, y])
        boundary_types.append('inner')

    ys = np.linspace(r, R, n_points_per_edge, endpoint=True)
    for y in ys:
        boundary_points.append([0, y])
        boundary_types.append('left')

    xs = np.linspace(r, R, n_points_per_edge, endpoint=True)
    for x in xs:
        boundary_points.append([x, 0])
        boundary_types.append('bottom')

    return np.array(boundary_points), np.array(boundary_types)


def import_data(i):
    workbook = xlrd.open_workbook('Target.xls')
    sheet = workbook.sheet_by_index(i)

    first_column = []

    for row_idx in range(sheet.nrows):
        first_column.append(sheet.cell_value(row_idx, 2))

    return first_column

' True value '

U_truth = import_data(0)
V_truth = import_data(1)
U_V_truth = import_data(2)
txx_truth = import_data(3)
tyy_truth = import_data(4)
txy_truth = import_data(5)
mise_truth = import_data(6)
Ux_truth = import_data(7)
Vy_truth = import_data(8)
Uxy_truth = import_data(9)
Uy_truth = import_data(10)
Vx_truth = import_data(11)

U_truth = np.array(U_truth, dtype=np.float32).reshape(-1, 1)
V_truth = np.array(V_truth, dtype=np.float32).reshape(-1, 1)
U_V_truth = np.array(U_V_truth, dtype=np.float32).reshape(-1, 1)
txx_truth = np.array(txx_truth, dtype=np.float32).reshape(-1, 1)
tyy_truth = np.array(tyy_truth, dtype=np.float32).reshape(-1, 1)
txy_truth = np.array(txy_truth, dtype=np.float32).reshape(-1, 1)
mise_truth = np.array(mise_truth, dtype=np.float32).reshape(-1, 1)
Ux_truth = np.array(Ux_truth, dtype=np.float32).reshape(-1, 1)
Vy_truth = np.array(Vy_truth, dtype=np.float32).reshape(-1, 1)
Uxy_truth = np.array(Uxy_truth, dtype=np.float32).reshape(-1, 1)
Vx_truth = np.array(Vx_truth, dtype=np.float32).reshape(-1, 1)
Uy_truth = np.array(Uy_truth, dtype=np.float32).reshape(-1, 1)

' Physical quantities '

E = 10.0
v = 0.30

' Generate coordinates '

R = 0.5
r = 0.1

nx = 30
ny = 30

r_list = np.linspace(r, R, nx).reshape(-1, 1).astype(np.float32)

theta_list = np.linspace(0, np.pi / 2, ny)

R_grid, Theta_grid = np.meshgrid(r_list, theta_list)

x_p = R_grid * np.cos(Theta_grid)
y_p = R_grid * np.sin(Theta_grid)

x_p = x_p.flatten().reshape(-1, 1)
y_p = y_p.flatten().reshape(-1, 1)

boundary_points, boundary_types = generate_boundary_points(R, r, ny)
boundary_x = boundary_points[:, 0].reshape(-1, 1)
boundary_y = boundary_points[:, 1].reshape(-1, 1)

outer_mask = boundary_types == 'outer'
outer_boundary_x = boundary_x[outer_mask]
outer_boundary_y = boundary_y[outer_mask]
outer_angles = np.arctan2(outer_boundary_y, outer_boundary_x)

inner_mask = boundary_types == 'inner'
inner_boundary_x = boundary_x[inner_mask]
inner_boundary_y = boundary_y[inner_mask]
inner_angles = np.arctan2(inner_boundary_y, inner_boundary_x)

left_mask = boundary_types == 'left'
bottom_mask = boundary_types == 'bottom'
inner_mask = boundary_types == 'inner'
outer_mask = boundary_types == 'outer'

left_indices = np.where(left_mask)[0].reshape(-1, 1)
bottom_indices = np.where(bottom_mask)[0].reshape(-1, 1)
inner_indices = np.where(inner_mask)[0].reshape(-1, 1)
outer_indices = np.where(outer_mask)[0].reshape(-1, 1)

' Obtain physical quantity '

load = 5.0 + 1.0 * np.sin(outer_angles)

# numpy to tensor
xp = tf.constant(x_p, dtype=tf.float32)
yp = tf.constant(y_p, dtype=tf.float32)

all_boundary_x = tf.constant(boundary_x, dtype=tf.float32)
all_boundary_y = tf.constant(boundary_y, dtype=tf.float32)

costheta_outer = tf.constant(np.cos(outer_angles), dtype=tf.float32)
sintheta_outer = tf.constant(np.sin(outer_angles), dtype=tf.float32)
load_outer = tf.constant(load, dtype=tf.float32)

costheta_inner = tf.constant(np.cos(inner_angles), dtype=tf.float32)
sintheta_inner = tf.constant(np.sin(inner_angles), dtype=tf.float32)

U_d, V_d, Sigma_x, Sigma_y, Sigma_xy, Ux, Vy, Uxy, Uy, Vx = PINN(xp, yp, reuse=False)

U_d_boundary, V_d_boundary, Sigma_x_boundary, Sigma_y_boundary, Sigma_xy_boundary, _, _, _, _, _ = PINN(all_boundary_x,
                                                                                                     all_boundary_y,
                                                                                                     reuse=True)
Sigma_x_inner = tf.gather_nd(Sigma_x_boundary, inner_indices)
Sigma_y_inner = tf.gather_nd(Sigma_y_boundary, inner_indices)
Sigma_xy_inner = tf.gather_nd(Sigma_xy_boundary, inner_indices)

Sigma_x_outer = tf.gather_nd(Sigma_x_boundary, outer_indices)
Sigma_y_outer = tf.gather_nd(Sigma_y_boundary, outer_indices)
Sigma_xy_outer = tf.gather_nd(Sigma_xy_boundary, outer_indices)

U_outer = tf.gather_nd(U_d_boundary, outer_indices)
V_outer = tf.gather_nd(V_d_boundary, outer_indices)

' Energy calculation '

dr = (R - r) / (nx - 1)
dtheta = (np.pi / 2) / (ny - 1)
area_weights = R_grid.flatten() * dr * dtheta
area_weights = area_weights.reshape(-1, 1)

boundary_weights = R * dtheta * np.ones_like(r_list).reshape(-1, 1)

weights = tf.constant(area_weights, dtype=tf.float32)
boundary_weights = tf.constant(boundary_weights, dtype=tf.float32)

l_inner = 0.5 * (Sigma_x * Ux + Sigma_y * Vy + Sigma_xy * Uxy) * weights # Please refer to Appendix C.2.
l_outer = (U_outer * load_outer * costheta_outer + V_outer * load * sintheta_outer) * boundary_weights

l = tf.reduce_sum(l_inner) - tf.reduce_sum(l_outer)

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 1e-3
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           1000, 0.95, staircase=True)
learning_rate = tf.maximum(learning_rate, 1e-6)
optimizer_P = tf.train.AdamOptimizer(learning_rate).minimize(l, global_step=global_step)

Stress_mises = tf.sqrt(0.5 * (tf.square(Sigma_x - Sigma_y) +
                              tf.square(Sigma_x) + tf.square(Sigma_y) +
                              6 * tf.square(Sigma_xy)))

U_V = tf.sqrt(tf.square(U_d) + tf.square(V_d))

start = time.perf_counter()

with tf.Session() as sess: # The fixed-point training is performed, and the final output is regarded as the prediction.
    sess.run(tf.global_variables_initializer())

    for i in range(100000):
        _, loss_val, lr_val = sess.run([optimizer_P, l, learning_rate])

        if i % 200 == 0:
            print(f"Iter {i}: Loss={loss_val:.2e}, lr={lr_val:.1e}")

    end = time.perf_counter()
    runTime = end - start
    runTime_ms = runTime * 1000
    print("运行时间：", runTime, "秒")
    print("运行时间：", runTime_ms, "毫秒")

    U_val, V_val, U_V_val, Stress_mises_val, Sigma_x_val, Sigma_y_val, Sigma_xy_val = sess.run([U_d, V_d, U_V, Stress_mises, Sigma_x, Sigma_y, Sigma_xy])

    U_V_res = abs(U_V_val - U_V_truth)
    Stress_mises_res = abs(Stress_mises_val - mise_truth)


    plt.figure(figsize=(10, 8))
    scatter1 = plt.scatter(x_p, y_p, c=U_val.flatten(), cmap='jet', s=20, alpha=0.7)

    cbar = plt.colorbar(scatter1)
    u_min = U_val.min()
    u_max = U_val.max()
    u_mid = (u_min + u_max) / 2

    cbar.set_ticks([u_min, u_mid, u_max])
    cbar.set_ticklabels([f'{u_min:.2e}', f'{u_mid:.2e}', f'{u_max:.2e}'])

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('U')
    plt.axis('equal')
    plt.show()

    plt.figure(figsize=(10, 8))
    scatter1 = plt.scatter(x_p, y_p, c=V_val.flatten(), cmap='jet', s=20, alpha=0.7)

    cbar = plt.colorbar(scatter1)
    v_min = V_val.min()
    v_max = V_val.max()
    v_mid = (v_min + v_max) / 2
    cbar.set_ticks([v_min, v_mid, v_max])
    cbar.set_ticklabels([f'{v_min:.2e}', f'{v_mid:.2e}', f'{v_max:.2e}'])

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('V')
    plt.axis('equal')
    plt.show()

    plt.figure(figsize=(10, 8))
    scatter1 = plt.scatter(x_p, y_p, c=U_V_val.flatten(), cmap='jet', s=20, alpha=0.7)

    cbar = plt.colorbar(scatter1)
    v_min = U_V_val.min()
    v_max = U_V_val.max()
    v_mid = (v_min + v_max) / 2
    cbar.set_ticks([v_min, v_mid, v_max])
    cbar.set_ticklabels([f'{v_min:.2e}', f'{v_mid:.2e}', f'{v_max:.2e}'])

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('U_V')
    plt.axis('equal')
    plt.show()

    plt.figure(figsize=(10, 8))
    scatter1 = plt.scatter(x_p, y_p, c=Sigma_x_val.flatten(), cmap='jet', s=20, alpha=0.7)

    cbar = plt.colorbar(scatter1)
    sigma_x_min = Sigma_x_val.min()
    sigma_x_max = Sigma_x_val.max()
    sigma_x_mid = (sigma_x_min + sigma_x_max) / 2
    cbar.set_ticks([sigma_x_min, sigma_x_mid, sigma_x_max])
    cbar.set_ticklabels([f'{sigma_x_min:.2e}', f'{sigma_x_mid:.2e}', f'{sigma_x_max:.2e}'])

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Sigma_x')
    plt.axis('equal')
    plt.show()

    plt.figure(figsize=(10, 8))
    scatter1 = plt.scatter(x_p, y_p, c=Sigma_y_val.flatten(), cmap='jet', s=20, alpha=0.7)

    cbar = plt.colorbar(scatter1)
    sigma_y_min = Sigma_y_val.min()
    sigma_y_max = Sigma_y_val.max()
    sigma_y_mid = (sigma_y_min + sigma_y_max) / 2
    cbar.set_ticks([sigma_y_min, sigma_y_mid, sigma_y_max])
    cbar.set_ticklabels([f'{sigma_y_min:.2e}', f'{sigma_y_mid:.2e}', f'{sigma_y_max:.2e}'])

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Sigma_y')
    plt.axis('equal')
    plt.show()

    plt.figure(figsize=(10, 8))
    scatter1 = plt.scatter(x_p, y_p, c=Sigma_xy_val.flatten(), cmap='jet', s=20, alpha=0.7)

    cbar = plt.colorbar(scatter1)
    sigma_xy_min = Sigma_xy_val.min()
    sigma_xy_max = Sigma_xy_val.max()
    sigma_xy_mid = (sigma_xy_min + sigma_xy_max) / 2
    cbar.set_ticks([sigma_xy_min, sigma_xy_mid, sigma_xy_max])
    cbar.set_ticklabels([f'{sigma_xy_min:.2e}', f'{sigma_xy_mid:.2e}', f'{sigma_xy_max:.2e}'])

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Sigma_xy')
    plt.axis('equal')
    plt.show()

    plt.figure(figsize=(10, 8))
    scatter1 = plt.scatter(x_p, y_p, c=Stress_mises_val.flatten(), cmap='jet', s=20, alpha=0.7)

    cbar = plt.colorbar(scatter1)
    v_min = Stress_mises_val.min()
    v_max = Stress_mises_val.max()
    v_mid = (v_min + v_max) / 2

    cbar.set_ticks([v_min, v_mid, v_max])
    cbar.set_ticklabels([f'{v_min:.2e}', f'{v_mid:.2e}', f'{v_max:.2e}'])

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Stress_mises')
    plt.axis('equal')
    plt.show()

    plt.figure(figsize=(10, 8))
    scatter1 = plt.scatter(x_p, y_p, c=U_V_res.flatten(), cmap='jet', s=20, alpha=0.7)

    cbar = plt.colorbar(scatter1)
    v_min = U_V_res.min()
    v_max = U_V_res.max()
    v_mid = (v_min + v_max) / 2

    cbar.set_ticks([v_min, v_mid, v_max])
    cbar.set_ticklabels([f'{v_min:.2e}', f'{v_mid:.2e}', f'{v_max:.2e}'])

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('U_V_error')
    plt.axis('equal')
    plt.show()

    plt.figure(figsize=(10, 8))
    scatter1 = plt.scatter(x_p, y_p, c=Stress_mises_res.flatten(), cmap='jet', s=20, alpha=0.7)

    cbar = plt.colorbar(scatter1)
    v_min = Stress_mises_res.min()
    v_max = Stress_mises_res.max()
    v_mid = (v_min + v_max) / 2

    cbar.set_ticks([v_min, v_mid, v_max])
    cbar.set_ticklabels([f'{v_min:.2e}', f'{v_mid:.2e}', f'{v_max:.2e}'])

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Stress_mises_error')
    plt.axis('equal')
    plt.show()