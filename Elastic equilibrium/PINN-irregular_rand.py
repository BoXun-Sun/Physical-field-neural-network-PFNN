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
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# tf.set_random_seed(1)

def generate_random_points_in_quarter_annulus(R, r, n_points):
    points = []
    while len(points) < n_points:
        x = np.random.uniform(0, R, 1)[0]
        y = np.random.uniform(0, R, 1)[0]
        dist = np.sqrt(x ** 2 + y ** 2)
        if r <= dist <= R and x >= 0 and y >= 0:
            points.append([x, y])
    return np.array(points)

def generate_boundary_points(R, r, n_theta):
    boundary_points = []
    boundary_types = []
    angles = np.linspace(0, np.pi / 2, n_theta)

    # Outer
    for theta in angles:
        boundary_points.append([R * np.cos(theta), R * np.sin(theta)])
        boundary_types.append('outer')
    # Inner
    for theta in angles:
        boundary_points.append([r * np.cos(theta), r * np.sin(theta)])
        boundary_types.append('inner')
    # Left
    ys = np.linspace(r, R, n_theta)
    for y in ys:
        boundary_points.append([0, y])
        boundary_types.append('left')
    # Bottom
    xs = np.linspace(r, R, n_theta)
    for x in xs:
        boundary_points.append([x, 0])
        boundary_types.append('bottom')

    return np.array(boundary_points), np.array(boundary_types)

def MLP1(x, reuse=None):
    with tf.variable_scope('MLnx', reuse=reuse):
        l1 = tf.layers.dense(x, 64)
        l1 = tf.nn.tanh(l1)
        l2 = tf.layers.dense(l1, 64)
        l2 = tf.nn.tanh(l2)
        l3 = tf.layers.dense(l2, 64)
        l3 = tf.nn.tanh(l3)
        output = tf.layers.dense(l3, 1)
    return output


def MLP2(x, reuse=None):
    with tf.variable_scope('MLny', reuse=reuse):
        l1 = tf.layers.dense(x, 64)
        l1 = tf.nn.tanh(l1)
        l2 = tf.layers.dense(l1, 64)
        l2 = tf.nn.tanh(l2)
        l3 = tf.layers.dense(l2, 64)
        l3 = tf.nn.tanh(l3)
        output = tf.layers.dense(l3, 1)
    return output


def PINN(x, y, reuse=None):
    with tf.GradientTape(persistent=True) as g:
        g.watch(x)
        g.watch(y)
        with tf.GradientTape(persistent=True) as gg:
            gg.watch(x)
            gg.watch(y)
            U = MLP1(tf.concat([x, y], axis=1), reuse=reuse) * x
            V = MLP2(tf.concat([x, y], axis=1), reuse=reuse) * y
        Ux = gg.gradient(U, x)
        Uy = gg.gradient(U, y)
        Vx = gg.gradient(V, x)
        Vy = gg.gradient(V, y)
        Uxy = Uy + Vx
        del gg
        txx = E / (1 - v * v) * (Ux + v * Vy)
        tyy = E / (1 - v * v) * (Vy + v * Ux)
        txy = E / 2 / (1 + v) * (Uxy)
    txxx = g.gradient(txx, x)
    txyy = g.gradient(txy, y)
    tyyy = g.gradient(tyy, y)
    txyx = g.gradient(txy, x)
    del g
    return U, V, txx, tyy, txy, txxx, tyyy, txyy, txyx, Ux, Uy, Vx, Vy, Uxy

def test():
    ' Test '

    nx_test, ny_test = 30, 30
    r_test = np.linspace(r, R, nx_test)
    theta_test = np.linspace(0, np.pi / 2, ny_test)
    R_grid, Theta_grid = np.meshgrid(r_test, theta_test)
    test_x_np = (R_grid * np.cos(Theta_grid)).flatten().reshape(-1, 1)
    test_y_np = (R_grid * np.sin(Theta_grid)).flatten().reshape(-1, 1)

    dr = (R - r) / (nx_test - 1)
    dtheta = (np.pi / 2) / (ny_test - 1)
    area_weights = R_grid.flatten() * dr * dtheta
    area_weights = area_weights.reshape(-1, 1)
    tf_weights = tf.constant(area_weights, dtype=tf.float32)

    test_x = tf.constant(test_x_np, dtype=tf.float32)
    test_y = tf.constant(test_y_np, dtype=tf.float32)

    test_U, test_V, test_txx, test_tyy, test_txy, _, _, _, _, test_Ux, test_Uy, test_Vx, test_Vy, test_Uxy = PINN(
        test_x, test_y, reuse=True)

    test_Stress_mises = tf.sqrt(
        0.5 * (tf.square(test_txx - test_tyy) + tf.square(test_txx) + tf.square(test_tyy) + 6 * tf.square(test_txy)))
    test_UV_mag = tf.sqrt(tf.square(test_U) + tf.square(test_V))

    return test_U, test_V, test_UV_mag, test_Stress_mises, test_txx, test_tyy, test_txy

def import_data(i):
    workbook = xlrd.open_workbook('Target.xls')
    sheet = workbook.sheet_by_index(i)
    first_column = []
    for row_idx in range(sheet.nrows):
        first_column.append(sheet.cell_value(row_idx, 2))
    return first_column

' True value '

U_truth = np.array(import_data(0), dtype=np.float32).reshape(-1, 1)
V_truth = np.array(import_data(1), dtype=np.float32).reshape(-1, 1)
U_V_truth = np.array(import_data(2), dtype=np.float32).reshape(-1, 1)
txx_truth = np.array(import_data(3), dtype=np.float32).reshape(-1, 1)
tyy_truth = np.array(import_data(4), dtype=np.float32).reshape(-1, 1)
txy_truth = np.array(import_data(5), dtype=np.float32).reshape(-1, 1)
mise_truth = np.array(import_data(6), dtype=np.float32).reshape(-1, 1)
Ux_truth = np.array(import_data(7), dtype=np.float32).reshape(-1, 1)
Vy_truth = np.array(import_data(8), dtype=np.float32).reshape(-1, 1)
Uxy_truth = np.array(import_data(9), dtype=np.float32).reshape(-1, 1)
Uy_truth = np.array(import_data(10), dtype=np.float32).reshape(-1, 1)
Vx_truth = np.array(import_data(11), dtype=np.float32).reshape(-1, 1)

tf_U_V_truth = tf.constant(U_V_truth, dtype=tf.float32)
tf_mise_truth = tf.constant(mise_truth, dtype=tf.float32)
tf_Ux_truth = tf.constant(Ux_truth, dtype=tf.float32)
tf_Uy_truth = tf.constant(Uy_truth, dtype=tf.float32)
tf_Vx_truth = tf.constant(Vx_truth, dtype=tf.float32)
tf_Vy_truth = tf.constant(Vy_truth, dtype=tf.float32)
tf_Uxy_truth = tf.constant(Uxy_truth, dtype=tf.float32)
tf_txx_truth = tf.constant(txx_truth, dtype=tf.float32)
tf_tyy_truth = tf.constant(tyy_truth, dtype=tf.float32)
tf_txy_truth = tf.constant(txy_truth, dtype=tf.float32)

' Physical quantities '

E = 10.0
v = 0.30

' Generate coordinates '

R = 0.5
r = 0.1

n_domain = 900 - 30 * 4
n_theta = 30

xy_p = generate_random_points_in_quarter_annulus(R, r, n_domain) # domain
x_train = xy_p[:, 0].reshape(-1, 1)
y_train = xy_p[:, 1].reshape(-1, 1)

boundary_points, boundary_types = generate_boundary_points(R, r, n_theta) # boundary
boundary_x = boundary_points[:, 0].reshape(-1, 1)
boundary_y = boundary_points[:, 1].reshape(-1, 1)

xp = tf.constant(x_train, dtype=tf.float32)
yp = tf.constant(y_train, dtype=tf.float32)

all_boundary_x = tf.constant(boundary_x, dtype=tf.float32)
all_boundary_y = tf.constant(boundary_y, dtype=tf.float32)

outer_mask = boundary_types == 'outer'
inner_mask = boundary_types == 'inner'
left_mask = boundary_types == 'left'
bottom_mask = boundary_types == 'bottom'

outer_indices = np.where(outer_mask)[0].reshape(-1, 1)
inner_indices = np.where(inner_mask)[0].reshape(-1, 1)
left_indices = np.where(left_mask)[0].reshape(-1, 1)
bottom_indices = np.where(bottom_mask)[0].reshape(-1, 1)

outer_angles = np.arctan2(boundary_y[outer_mask], boundary_x[outer_mask])
inner_angles = np.arctan2(boundary_y[inner_mask], boundary_x[inner_mask])

' Obtain physical quantity '

load_outer_val = 5.0 + 1.0 * np.sin(outer_angles)

costheta_outer = tf.constant(np.cos(outer_angles), dtype=tf.float32)
sintheta_outer = tf.constant(np.sin(outer_angles), dtype=tf.float32)
load_outer = tf.constant(load_outer_val, dtype=tf.float32)
costheta_inner = tf.constant(np.cos(inner_angles), dtype=tf.float32)
sintheta_inner = tf.constant(np.sin(inner_angles), dtype=tf.float32)

_, _, _, _, _, Sigma_xx, Sigma_yy, Sigma_xyy, Sigma_xyx, _, _, _, _, _ = PINN(xp, yp, reuse=False)

U_b, V_b, Sigma_x_b, Sigma_y_b, Sigma_xy_b, _, _, _, _, _, _, _, _, _ = PINN(all_boundary_x, all_boundary_y, reuse=True)

Sigma_x_inner = tf.gather_nd(Sigma_x_b, inner_indices)
Sigma_y_inner = tf.gather_nd(Sigma_y_b, inner_indices)
Sigma_xy_inner = tf.gather_nd(Sigma_xy_b, inner_indices)

Sigma_x_outer = tf.gather_nd(Sigma_x_b, outer_indices)
Sigma_y_outer = tf.gather_nd(Sigma_y_b, outer_indices)
Sigma_xy_outer = tf.gather_nd(Sigma_xy_b, outer_indices)

U_left = tf.gather_nd(U_b, left_indices)
Sigma_xy_left = tf.gather_nd(Sigma_xy_b, left_indices)

V_bottom = tf.gather_nd(V_b, bottom_indices)
Sigma_xy_bottom = tf.gather_nd(Sigma_xy_b, bottom_indices)

' loss '

# equilibrium
l1 = tf.reduce_mean(tf.square(Sigma_xx + Sigma_xyy)) + tf.reduce_mean(tf.square(Sigma_yy + Sigma_xyx))

# left
l2 = tf.reduce_mean(tf.square(U_left)) + tf.reduce_mean(tf.square(Sigma_xy_left))

# bottom
l3 = tf.reduce_mean(tf.square(V_bottom)) + tf.reduce_mean(tf.square(Sigma_xy_bottom))

# outside
l41 = (Sigma_x_outer * costheta_outer ** 2 + Sigma_y_outer * sintheta_outer ** 2 + 2 * Sigma_xy_outer * sintheta_outer * costheta_outer) - load_outer
l42 = ((Sigma_y_outer - Sigma_x_outer) * sintheta_outer * costheta_outer + Sigma_xy_outer * (costheta_outer ** 2 - sintheta_outer ** 2))
l4 = tf.reduce_mean(tf.square(l41)) + tf.reduce_mean(tf.square(l42))

# inner
l51 = (Sigma_x_inner * costheta_inner ** 2 + Sigma_y_inner * sintheta_inner ** 2 + 2 * Sigma_xy_inner * sintheta_inner * costheta_inner)
l52 = ((Sigma_y_inner - Sigma_x_inner) * sintheta_inner * costheta_inner + Sigma_xy_inner * (costheta_inner ** 2 - sintheta_inner ** 2))
l5 = tf.reduce_mean(tf.square(l51)) + tf.reduce_mean(tf.square(l52))

loss = l1 + l2 + l3 + l4 + l5

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(1e-3, global_step, 1000, 0.95, staircase=True)
learning_rate = tf.maximum(learning_rate, 1e-6)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

start = time.perf_counter()

with tf.Session() as sess:
    'Train'

    sess.run(tf.global_variables_initializer())

    for i in range(100000):
        _, loss_val, lr_val = sess.run([optimizer, loss, learning_rate])

        if i % 200 == 0:

            l1_val, l2_val, l3_val, l4_val, l5_val = sess.run([l1, l2, l3, l4, l5])

            print(f"Iter {i}: Loss={loss_val:.2e}, lr={lr_val:.1e}")
            print(f"lg={l1_val:.2e}, l_left={l2_val:.2e}, l_bottom={l3_val:.2e}, l_bottom={l4_val:.2e}, l_outside={l5_val:.2e}")

    end = time.perf_counter()
    runtime = end - start
    print(f"训练时间: {runtime:.2f} 秒")

    'Prediction'
    U_val, V_val, U_V_val, Stress_mises_val, Sigma_x_val, Sigma_y_val, Sigma_xy_val = sess.run(test())

    U_V_res = abs(U_V_val - U_V_truth)
    Stress_mises_res = abs(Stress_mises_val - mise_truth)

    nx_test, ny_test = 30, 30
    r_test = np.linspace(r, R, nx_test)
    theta_test = np.linspace(0, np.pi / 2, ny_test)
    R_grid, Theta_grid = np.meshgrid(r_test, theta_test)
    test_x_np = (R_grid * np.cos(Theta_grid)).flatten().reshape(-1, 1)
    test_y_np = (R_grid * np.sin(Theta_grid)).flatten().reshape(-1, 1)

    plt.figure(figsize=(10, 8))
    scatter1 = plt.scatter(test_x_np, test_y_np, c=U_val.flatten(), cmap='jet', s=20, alpha=0.7)

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
    scatter1 = plt.scatter(test_x_np, test_y_np, c=V_val.flatten(), cmap='jet', s=20, alpha=0.7)

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
    scatter1 = plt.scatter(test_x_np, test_y_np, c=U_V_val.flatten(), cmap='jet', s=20, alpha=0.7)

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
    scatter1 = plt.scatter(test_x_np, test_y_np, c=Sigma_x_val.flatten(), cmap='jet', s=20, alpha=0.7)

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
    scatter1 = plt.scatter(test_x_np, test_y_np, c=Sigma_y_val.flatten(), cmap='jet', s=20, alpha=0.7)

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
    scatter1 = plt.scatter(test_x_np, test_y_np, c=Sigma_xy_val.flatten(), cmap='jet', s=20, alpha=0.7)

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
    scatter1 = plt.scatter(test_x_np, test_y_np, c=Stress_mises_val.flatten(), cmap='jet', s=20, alpha=0.7)

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
    scatter1 = plt.scatter(test_x_np, test_y_np, c=U_V_res.flatten(), cmap='jet', s=20, alpha=0.7)

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
    scatter1 = plt.scatter(test_x_np, test_y_np, c=Stress_mises_res.flatten(), cmap='jet', s=20, alpha=0.7)

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