'''
    Elastic equilibrium equation
    Please refer to Section 3.3.
    If you have any question, please don’t hesitate to contact us (23121084@bjtu.edu.cn).
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

def FDFDs1(U, choose):
    if choose == 0:

        # First-order derivative with respect to y
        front = (-3 * U[0] + 4 * U[1] - U[2]) / (2 * long_ry)
        center = (U[2:] - U[:-2]) / (2 * long_ry)
        back = (3 * U[-1] - 4 * U[-2] + U[-3]) / (2 * long_ry)

        U_x = tf.concat([
            front[None, :],
            center,
            back[None, :]
        ], axis=0)

    else:

        # First-order derivative with respect to x
        U_T = tf.transpose(U)
        front = (-3 * U_T[0] + 4 * U_T[1] - U_T[2]) / (2 * long_rx)
        center = (U_T[2:] - U_T[:-2]) / (2 * long_rx)
        back = (3 * U_T[-1] - 4 * U_T[-2] + U_T[-3]) / (2 * long_rx)

        U_x = tf.transpose(tf.concat([
            front[None, :],
            center,
            back[None, :]
        ], axis=0))

    return tf.reshape(U_x, [-1, 1])


def FDFDs1_np(U, choose):

    U = U.reshape(nx, ny)

    if choose == 0:
        front = (-3 * U[0, :] + 4 * U[1, :] - U[2, :]) / (2 * long_ry)
        center = (U[2:, :] - U[:-2, :]) / (2 * long_ry)
        back = (3 * U[-1, :] - 4 * U[-2, :] + U[-3, :]) / (2 * long_ry)

        U_x = np.vstack([
            front[np.newaxis, :],
            center,
            back[np.newaxis, :]
        ])
    else:
        front = (-3 * U[:, 0] + 4 * U[:, 1] - U[:, 2]) / (2 * long_rx)
        center = (U[:, 2:] - U[:, :-2]) / (2 * long_rx)
        back = (3 * U[:, -1] - 4 * U[:, -2] + U[:, -3]) / (2 * long_rx)
        U_x = np.hstack([
            front[:, np.newaxis],
            center,
            back[:, np.newaxis]
        ])

    return U_x.reshape(-1, 1)


def MLnx(x):
    with tf.variable_scope('MLnx'):
        l1 = tf.layers.dense(x, 64)
        l1 = tf.nn.relu(l1)
        l2 = tf.layers.dense(l1, 64)
        l2 = tf.nn.relu(l2)
        l3 = tf.layers.dense(l2, 64)
        l3 = tf.nn.relu(l3)
        output = tf.layers.dense(l3, nx * ny)
    return output


def MLny(x):
    with tf.variable_scope('MLny'):
        l1 = tf.layers.dense(x, 64)
        l1 = tf.nn.relu(l1)
        l2 = tf.layers.dense(l1, 64)
        l2 = tf.nn.relu(l2)
        l3 = tf.layers.dense(l2, 64)
        l3 = tf.nn.relu(l3)
        output = tf.layers.dense(l3, nx * ny)
    return output


def PINN():
    U = MLnx(xy) * xp
    V = MLny(xy) * yp

    U_d = tf.reshape(U, [ny, nx])
    V_d = tf.reshape(V, [ny, nx])
    U_d_xi = FDFDs1(tf.reshape(U_d, [ny, nx]), 1)  # First-order derivative with respect to x
    U_d_deta = FDFDs1(tf.reshape(U_d, [ny, nx]), 0)  # First-order derivative with respect to y
    V_d_xi = FDFDs1(tf.reshape(V_d, [ny, nx]), 1)  # First-order derivative with respect to x
    V_d_deta = FDFDs1(tf.reshape(V_d, [ny, nx]), 0)  # First-order derivative with respect to y

    Ux = J_inv * (U_d_xi * dydeta - U_d_deta * dydxi)
    Uy = J_inv * (U_d_deta * dxdxi - U_d_xi * dxdeta)

    Vx = J_inv * (V_d_xi * dydeta - V_d_deta * dydxi)
    Vy = J_inv * (V_d_deta * dxdxi - V_d_xi * dxdeta)

    Sigma_x = 1.0 / (1 - v * v) * (Ux + v * Vy)
    Sigma_y = 1.0 / (1 - v * v) * (Vy + v * Ux)
    Sigma_xy = 1.0 / 2 / (1 + v) * (Vx + Uy)

    Sigma_x_xi = FDFDs1(tf.reshape(Sigma_x, [ny, nx]), 1)  # First-order derivative with respect to x
    Sigma_x_deta = FDFDs1(tf.reshape(Sigma_x, [ny, nx]), 0)  # First-order derivative with respect to y
    Sigma_y_xi = FDFDs1(tf.reshape(Sigma_y, [ny, nx]), 1)  # First-order derivative with respect to x
    Sigma_y_deta = FDFDs1(tf.reshape(Sigma_y, [ny, nx]), 0)  # First-order derivative with respect to y
    Sigma_xy_xi = FDFDs1(tf.reshape(Sigma_xy, [ny, nx]), 1)  # First-order derivative with respect to x
    Sigma_xy_deta = FDFDs1(tf.reshape(Sigma_xy, [ny, nx]), 0)  # First-order derivative with respect to y

    Sigma_xx = J_inv * (Sigma_x_xi * dydeta - Sigma_x_deta * dydxi)
    Sigma_yy = J_inv * (Sigma_y_deta * dxdxi - Sigma_y_xi * dxdeta)

    Sigma_xyx = J_inv * (Sigma_xy_xi * dydeta - Sigma_xy_deta * dydxi)
    Sigma_xyy = J_inv * (Sigma_xy_deta * dxdxi - Sigma_xy_xi * dxdeta)

    U_d = tf.reshape(U_d, [-1, 1])
    V_d = tf.reshape(V_d, [-1, 1])

    return U_d, V_d, Sigma_x * E, Sigma_y * E, Sigma_xy * E, Sigma_xx * E, Sigma_yy * E, Sigma_xyy * E, Sigma_xyx * E, Ux, Uy, Vx, Vy, Vx + Uy


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

nx = 30
ny = 30

R = 0.5
r = 0.1

xr_s = 0.0 # regular domain
xr_l = 0.5
yr_s = 0.0
yr_l = 0.5

r_list = np.linspace(r, R, nx).reshape(-1, 1).astype(np.float32)

long = r_list[1] - r_list[0]

theta_list = np.linspace(0, np.pi / 2, ny)

xy_p = [] # irregular domain

for i in theta_list:
    for j in r_list:
        x_p = j * np.cos(i)
        y_p = j * np.sin(i)
        xy_p.append([x_p, y_p])

xy_p = np.array(xy_p).reshape(-1, 2)

x_p = xy_p[:, 0]
y_p = xy_p[:, 1]

x_p = x_p.reshape(-1, 1)
y_p = y_p.reshape(-1, 1)

costheta = np.cos(theta_list)
sintheta = np.sin(theta_list)


x_list = np.linspace(xr_s, xr_l, nx).reshape(-1, 1).astype(np.float32)
y_list = np.linspace(yr_s, yr_l, ny).reshape(-1, 1).astype(np.float32)

long_rx = x_list[1] - x_list[0]
long_ry = y_list[1] - y_list[0]

x_r, y_r = np.meshgrid(x_list, y_list)

x_r = x_r.reshape(-1, 1)
y_r = y_r.reshape(-1, 1)

tolerance = 1e-6

xy_r = np.vstack([x_r.ravel(), y_r.ravel()]).T

inner_boundary_indices = np.where(np.abs(xy_r[:, 0] - xr_s) < tolerance)[0].reshape(-1, 1)
outside_boundary_indices = np.where(np.abs(xy_r[:, 0] - xr_l) < tolerance)[0].reshape(-1, 1)
bottom_boundary_indices = np.where(np.abs(xy_r[:, 1] - yr_s) < tolerance)[0].reshape(-1, 1)
left_boundary_indices = np.where(np.abs(xy_r[:, 1] - yr_l) < tolerance)[0].reshape(-1, 1)

dxdxi = FDFDs1_np(x_p, 1)  # Coordinate transformation. Please refer to the seconda paragraph in Section 3.3, and Appendix A.6
dxdeta = FDFDs1_np(x_p, 0)
dydxi = FDFDs1_np(y_p, 1)
dydeta = FDFDs1_np(y_p, 0)

J = dxdxi * dydeta - dxdeta * dydxi # The determinant of the Jacobian matrix

J_inv = 1 / J

xr = tf.reshape(tf.constant(x_r), [1, -1])
yr = tf.reshape(tf.constant(y_r), [1, -1])

xp = tf.reshape(tf.constant(x_p), [1, -1])
yp = tf.reshape(tf.constant(y_p), [1, -1])

v1 = tf.Variable(tf.ones_like(xr, dtype=tf.float32))
v2 = tf.Variable(tf.ones_like(yr, dtype=tf.float32))

costheta = tf.reshape(tf.constant(costheta, dtype=tf.float32), [-1, 1])
sintheta = tf.reshape(tf.constant(sintheta, dtype=tf.float32), [-1, 1])

xy = xr * v1 + yr * v2
xy = tf.concat(xy, 1)

' Obtain physical quantity '

U_d, V_d, Sigma_x, Sigma_y, Sigma_xy, Sigma_xx, Sigma_yy, Sigma_xyy, Sigma_xyx, Ux, Uy, Vx, Vy, UV_xy = PINN()

Sigma_x_inner = tf.gather_nd(Sigma_x, inner_boundary_indices)
Sigma_y_inner = tf.gather_nd(Sigma_y, inner_boundary_indices)
Sigma_xy_inner = tf.gather_nd(Sigma_xy, inner_boundary_indices)

Sigma_x_outside = tf.gather_nd(Sigma_x, outside_boundary_indices)
Sigma_y_outside = tf.gather_nd(Sigma_y, outside_boundary_indices)
Sigma_xy_outside = tf.gather_nd(Sigma_xy, outside_boundary_indices)

load = 5.0 + 1.0 * sintheta

' loss '

# equilibrium

l11 = Sigma_xx + Sigma_xyy
l12 = Sigma_yy + Sigma_xyx
l1 = tf.reduce_mean(tf.square(l11 / E)) + tf.reduce_mean(tf.square(l12 / E))

# left
l21 = tf.reshape(tf.gather_nd(U_d, left_boundary_indices), [-1, 1])
l22 = tf.reshape(tf.gather_nd(Sigma_xy, left_boundary_indices), [-1, 1])
l2 = tf.reduce_mean(tf.square(l21)) + tf.reduce_mean(tf.square(l22))

# outside
l31 = (Sigma_x_outside * costheta * costheta + Sigma_y_outside * sintheta * sintheta + 2 * Sigma_xy_outside * sintheta * costheta) - load
l32 = ((Sigma_y_outside - Sigma_x_outside) * sintheta * costheta + Sigma_xy_outside * (costheta ** 2 - sintheta ** 2))
l3 = tf.reduce_mean(tf.square(l31)) + tf.reduce_mean(tf.square(l32))

# bottom
l41 = tf.reshape(tf.gather_nd(V_d, bottom_boundary_indices), [-1, 1])
l42 = tf.reshape(tf.gather_nd(Sigma_xy, bottom_boundary_indices), [-1, 1])
l4 = tf.reduce_mean(tf.square(l41)) + tf.reduce_mean(tf.square(l42))

# inner
l51 = (Sigma_x_inner * costheta * costheta + Sigma_y_inner * sintheta * sintheta + 2 * Sigma_xy_inner * sintheta * costheta)
l52 = ((Sigma_y_inner - Sigma_x_inner) * sintheta * costheta + Sigma_xy_inner * (costheta ** 2 - sintheta ** 2))
l5 = tf.reduce_mean(tf.square(l51)) + tf.reduce_mean(tf.square(l52))

l = l1 + l2 + l3 + l4 + l5

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 1e-3
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           1000, 0.95, staircase=True) # According to extensive tests
learning_rate = tf.maximum(learning_rate, 1e-6)
optimizer_P = tf.train.AdamOptimizer(learning_rate).minimize(l, global_step=global_step)

U_V = tf.sqrt(tf.square(U_d) + tf.square(V_d))

Stress_mises = tf.sqrt(0.5 * (tf.square(Sigma_x - Sigma_y) +
                              tf.square(Sigma_x) + tf.square(Sigma_y) +
                              6 * tf.square(Sigma_xy)))

start = time.perf_counter()

with tf.Session() as sess: # The fixed-point training is performed, and the final output is regarded as the prediction.
    sess.run(tf.global_variables_initializer())

    for i in range(100000):

        _, loss_val, lr_val = sess.run([optimizer_P, l, learning_rate])

        if i % 200 == 0:

            l1_val, l2_val, l3_val, l4_val, l5_val = sess.run([l1, l2, l3, l4, l5])
            print(f"Iter {i}: Loss={loss_val:.2e}, lr={lr_val:.1e}")
            print(f"lg={l1_val:.2e}, l_left={l2_val:.2e}, l_outside={l3_val:.2e}, l_bottom={l4_val:.2e}, l_inner={l5_val:.2e}")

    print(f"lg={l1_val:.2e}, l_left={l2_val:.2e}, l_outside={l3_val:.2e}, l_bottom={l4_val:.2e}, l_inner={l5_val:.2e}")  # Due to uniform sampling, the output can be directly regarded as the prediction

    end = time.perf_counter()
    runTime = end - start
    print("运行时间：", runTime, "秒")

    U_V_val, U_val, V_val, Stress_mises_val, Sigma_x_val, Sigma_y_val, Sigma_xy_val = sess.run(
        [U_V, U_d, V_d, Stress_mises, Sigma_x, Sigma_y, Sigma_xy])


    U_V_res = abs(U_V_val - U_V_truth)
    Stress_mises_res = abs(Stress_mises_val - mise_truth)

    plt.figure(figsize=(10, 8))
    scatter1 = plt.scatter(xy_p[:, 0], xy_p[:, 1], c=U_val.flatten(), cmap='jet', s=20, alpha=0.7)

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
    scatter1 = plt.scatter(xy_p[:, 0], xy_p[:, 1], c=V_val.flatten(), cmap='jet', s=20, alpha=0.7)

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
    scatter1 = plt.scatter(xy_p[:, 0], xy_p[:, 1], c=U_V_val.flatten(), cmap='jet', s=20, alpha=0.7)

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
    scatter1 = plt.scatter(xy_p[:, 0], xy_p[:, 1], c=Sigma_x_val.flatten(), cmap='jet', s=20, alpha=0.7)

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
    scatter1 = plt.scatter(xy_p[:, 0], xy_p[:, 1], c=Sigma_y_val.flatten(), cmap='jet', s=20, alpha=0.7)

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
    scatter1 = plt.scatter(xy_p[:, 0], xy_p[:, 1], c=Sigma_xy_val.flatten(), cmap='jet', s=20, alpha=0.7)

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
    scatter1 = plt.scatter(xy_p[:, 0], xy_p[:, 1], c=Stress_mises_val.flatten(), cmap='jet', s=20, alpha=0.7)

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
    scatter1 = plt.scatter(xy_p[:, 0], xy_p[:, 1], c=U_V_res.flatten(), cmap='jet', s=20, alpha=0.7)

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
    scatter1 = plt.scatter(xy_p[:, 0], xy_p[:, 1], c=Stress_mises_res.flatten(), cmap='jet', s=20, alpha=0.7)

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
