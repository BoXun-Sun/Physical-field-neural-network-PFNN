'''
    Electro-Thermal equation
    Please refer to Section 3.4.
    If you have any question, please don’t hesitate to contact us (23121084@bjtu.edu.cn).
    '''

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import xlrd
import os
import matplotlib.pylab as plt
import time
from matplotlib.ticker import FuncFormatter
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # run the code with the CPU

# tf.set_random_seed(2)

def sci_fmt(x, pos):
    return f'{x:.1e}'

def MLP1(x): # Please refer to Section 2.2
    with tf.variable_scope('MLP1'):
        l1 = tf.layers.dense(x, 32)
        l1 = tf.nn.relu(l1)
        l2 = tf.layers.dense(l1, 32)
        l2 = tf.nn.relu(l2)
        l3 = tf.layers.dense(l2, 32)
        l3 = tf.nn.relu(l3)
        output = tf.layers.dense(l3, p1*p2)
    return output

def MLP2(x): # Please refer to Section 2.2
    with tf.variable_scope('MLP2'):
        l1 = tf.layers.dense(x, 32)
        l1 = tf.nn.relu(l1)
        l2 = tf.layers.dense(l1, 32)
        l2 = tf.nn.relu(l2)
        l3 = tf.layers.dense(l2, 32)
        l3 = tf.nn.relu(l3)
        output = tf.layers.dense(l3, p1*p2)
    return output

def coordinat():
    xye = []
    lflag = []
    rflag = []
    bflag = []
    uflag = []
    for j in range(0, p2):
        for i in range(0, p1):
            xye.append([x_list[i], y_list[j]])
            if i == 0:
                lflag.append([x_list[i], y_list[j]])
            if i == p1 - 1:
                rflag.append([x_list[i], y_list[j]])
            if j == 0:
                bflag.append([x_list[i], y_list[j]])
            if j == p2 - 1:
                uflag.append([x_list[i], y_list[j]])
    xye = np.array(xye).reshape(p1*p2, 2)
    bindexu = (np.where((xye[:, None] == uflag).all(-1))[0]).reshape(p1, 1)
    bindexb = (np.where((xye[:, None] == bflag).all(-1))[0]).reshape(p1, 1)
    bindexl = (np.where((xye[:, None] == lflag).all(-1))[0]).reshape(p2, 1)
    bindexr = (np.where((xye[:, None] == rflag).all(-1))[0]).reshape(p2, 1)
    return xye, bindexu, bindexb, bindexl, bindexr

def FDFDs1(U, choose):
    if choose == 0:
    # First-order derivative with respect to y
        front = (-3 * U[0] + 4 * U[1] - U[2]) / (2 * long)
        center = (U[2:] - U[:-2]) / (2 * long)
        back = (3 * U[-1] - 4 * U[-2] + U[-3]) / (2 * long)
        U_x = tf.concat([
            front[None, :],
            center,
            back[None, :]
        ], axis=0)
    else:
    # First-order derivative with respect to x
        U_T = tf.transpose(U)
        front = (-3 * U_T[0] + 4 * U_T[1] - U_T[2]) / (2 * long)
        center = (U_T[2:] - U_T[:-2]) / (2 * long)
        back = (3 * U_T[-1] - 4 * U_T[-2] + U_T[-3]) / (2 * long)
        U_x = tf.transpose(tf.concat([
            front[None, :],
            center,
            back[None, :]
        ], axis=0))

    return tf.reshape(U_x, [-1, 1])

def FDFDs2(U, choose):
    long_sq = long ** 2
    if choose == 0:
        # Second-order derivative with respect to y
        front = (2 * U[0] - 5 * U[1] + 4 * U[2] - U[3]) / long_sq
        center = (U[2:] - 2 * U[1:-1] + U[:-2]) / long_sq
        back = (2 * U[-1] - 5 * U[-2] + 4 * U[-3] - U[-4]) / long_sq
        U_xx = tf.concat([
            front[None, :],
            center,
            back[None, :]
        ], axis=0)
    else:
        # Second-order derivative with respect to x
        U_T = tf.transpose(U)
        front = (2 * U_T[0] - 5 * U_T[1] + 4 * U_T[2] - U_T[3]) / long_sq
        center = (U_T[2:] - 2 * U_T[1:-1] + U_T[:-2]) / long_sq
        back = (2 * U_T[-1] - 5 * U_T[-2] + 4 * U_T[-3] - U_T[-4]) / long_sq
        U_xx = tf.transpose(tf.concat([
            front[None, :],
            center,
            back[None, :]
        ], axis=0))
    return tf.reshape(U_xx, [p1*p2,1])

def PINN1():
    Ur = MLP1(xy1) * (0.5*0.5 - y * y) + (0.5 + y)   # Output physical field
    Ur = tf.reshape(Ur, [p2 * p1, 1])
    U = Ur
    U = tf.reshape(U, [p2, p1])
    Ux = FDFDs1(U, 1)   # First-order derivative with respect to x
    Uy = FDFDs1(U, 0)   # First-order derivative with respect to y
    Uxx = FDFDs2(U, 1)  # Second-order derivative with respect to x
    Uyy = FDFDs2(U, 0)
    U = tf.reshape(U, [p1*p2, 1])
    return U, Ux, Uy, Uxx, Uyy

def PINN2():
    Ur = MLP2(xy2) * (x*x - 0.5*0.5) * (y*y - 0.5*0.5) + 273.0  # Output physical field
    Ur = tf.reshape(Ur, [p2 * p1, 1])
    U = Ur
    U = tf.reshape(U, [p2, p1])
    Uxx = FDFDs2(U, 1)  # Second-order derivative with respect to x
    Uyy = FDFDs2(U, 0)
    U = tf.reshape(U, [p1*p2, 1])
    return U, Uxx, Uyy

def H1(Ut, U):

    U_res = U - Ut

    U_res = tf.reshape(U_res, [p2, p1])

    Ux = FDFDs1(tf.reshape(U_res, [p2, p1]), 1)
    Uy = FDFDs1(tf.reshape(U_res, [p2, p1]), 0)

    H1 = tf.sqrt(tf.reduce_sum((Ux * Ux + Uy * Uy) * long * long))

    return H1

def MAE(Ut, U):

    U_res = U - Ut

    return tf.reduce_mean(tf.abs(U_res)), U_res, tf.abs(U_res)

def lg2(Ut, U):

    U_res = (U - Ut)

    return tf.sqrt(tf.reduce_sum((tf.square(U_res)) * long  * long))


def import_data(filename, sheet_name, cloumn1, cloumn2, cloumn3):

    workbook = xlrd.open_workbook(filename)
    sheet = workbook.sheet_by_name(sheet_name)

    first_column = []
    second_column = []
    third_column = []
    for row_idx in range(sheet.nrows):

        first_column.append(sheet.cell_value(row_idx, cloumn1))
        second_column.append(sheet.cell_value(row_idx, cloumn2))
        third_column.append(sheet.cell_value(row_idx, cloumn3))

    return first_column, second_column, third_column

U_truth, T_truth, _ = import_data('electro-thermal-data.xls', 'Sheet1', 3, 4, 0)

# _, T_data, T_indice = import_data('Impluse-noise.xls', 'T-15', 1, 1, 0)

_, T_data, T_indice = import_data('Gaussion-noise.xls', 'T-15', 1, 1, 0)

U_truth = np.array(U_truth, dtype=np.float32).reshape(-1, 1)
T_truth = np.array(T_truth, dtype=np.float32).reshape(-1, 1)

T_data = np.array(T_data).reshape(-1, 1)
T_indice = np.int_(np.array(T_indice).reshape(-1, 1))

T_data = tf.constant(T_data, dtype=tf.float32)

'Generate coordinate'

xs = -0.5  # Minimum x-coordinate
xl = 0.5   # Maximum x-coordinate
ys = -0.5  # Minimum y-coordinate
yl = 0.5  # Maximum y-coordinate

p1 = 50  # Number of collocation points along the x-direction
p2 = 50  # Number of collocation points along the y-direction

x_list = np.linspace(xs, xl, p1).astype(np.float32)
y_list = np.linspace(ys, yl, p2).astype(np.float32)

long = x_list[1] - x_list[0]  # Step size

xye, bindexu, bindexb, bindexl, bindexr = coordinat()  # Generate coordinate, where xye denote a matrix of whole coordinates,\
                                                       # bindexu, bindexb, bindexl, bindexr are the marker vectors for the top, bottom, left, and right boundaries, respectively.

x, y = (xye[..., i, tf.newaxis] for i in range(xye.shape[-1]))

x = tf.reshape(tf.constant(x), [1, p1*p1])
y = tf.reshape(tf.constant(y), [1, p1*p1])

' Encoding '

v11 = tf.Variable(tf.ones_like(x, dtype=tf.float32))  # Encoding coefficient matrix for x
v12 = tf.Variable(tf.ones_like(y, dtype=tf.float32))  # Encoding coefficient matrix for y

v21 = tf.Variable(tf.ones_like(x, dtype=tf.float32))  # Encoding coefficient matrix for x
v22 = tf.Variable(tf.ones_like(y, dtype=tf.float32))  # Encoding coefficient matrix for y

xy1 = x * v11 + y * v12  # Encoding
xy1 = tf.concat(xy1, 1)

xy2 = x * v21 + y * v22  # Encoding
xy2 = tf.concat(xy2, 1)

' Obtain physical quantity '

U, Ux, Uy, Uxx, Uyy = PINN1()
T, Txx, Tyy = PINN2()

' Loss '

# sigma_v = tf.Variable(tf.ones([1, 1], tf.float32))
k_v = tf.Variable(tf.ones([1, 1], tf.float32))

k = tf.square(k_v)

# k = 1/500
sigma = 1.0

E_x = -Ux
E_y = -Uy
J_x = sigma * E_x
J_y = sigma * E_y
Q = J_x * E_x + J_y * E_y

lg1i = sigma * (Uxx + Uyy)
lg1 = tf.reduce_mean(tf.square(lg1i))

lg2i = k * (Txx + Tyy) + Q
lg2 = tf.reduce_mean(tf.square(lg2i))

lb1i = tf.gather_nd(Ux, bindexl)
lb1 = tf.reduce_mean(tf.square(lb1i))

lb2i = tf.gather_nd(Ux, bindexr)
lb2 = tf.reduce_mean(tf.square(lb2i))

ldi = tf.gather_nd(T, T_indice) - T_data
ld = tf.reduce_mean(tf.square(ldi))

l = lg1 + lg2 + lb1 + lb2 + 0.1 * ld

' Optimization '

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 1e-4
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,20000, 0.5, staircase=True) # According to extensive tests
optimizer_P = tf.train.AdamOptimizer(learning_rate).minimize(l, global_step=global_step)

start = time.perf_counter()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    k_list = []
    iter_list = []

    for i in range(50000):

        _, loss_val, lr_val, k_val = sess.run([optimizer_P, l, learning_rate, k])
        k_list.append(k_val.flat[0])
        iter_list.append(i)

        if i % 500 == 0:
            lg1_val, lg2_val, lb1_val, lb2_val, ld_val = sess.run([lg1, lg2, lb1, lb2, ld])
            print(f"Iter {i}: Loss={loss_val:.2e}, lg1={lg1_val:.2e}, lg2={lg2_val:.2e}, lb1={lb1_val:.2e}, lb2={lb2_val:.2e}, ld={ld_val:.2e}")
            print(f"k_val={k_val.flat[0]:.2e}")

    print(f"Iter {i}: Loss={loss_val:.2e}, lg1={lg1_val:.2e}, lg2={lg2_val:.2e}, lb1={lb1_val:.2e}, lb2={lb2_val:.2e}, ld={ld_val:.2e}")
    print(f"k_val={k_val.flat[0]:.2e}")

    end = time.perf_counter()
    runTime = end - start
    print("RunTime：", runTime, "seconds")

    U_val, T_val = sess.run([U, T])

U_res = abs(U_val - U_truth)
T_res = abs(T_val - T_truth)

xc, yc = np.meshgrid(x_list, y_list)

fig = plt.figure(figsize=(6, 4))
plt.plot(iter_list, k_list, label='True', color='red', linestyle='-', linewidth=2, zorder=1)
plt.xlabel("Iter", size=15)
plt.ylabel("W/(m · K)", size=15)
plt.ylim(top=1.5)
plt.legend(loc='upper left', frameon=False, framealpha=0, fontsize=12)
plt.show()

fig = plt.figure(figsize=(6, 4))
plt.pcolormesh(xc, yc, U_val.reshape(xc.shape), shading='auto', cmap=plt.cm.rainbow)
plt.gca().ticklabel_format(style='scientific', scilimits=(-0.5, 0.5), useMathText=True)
clb = plt.colorbar()
clb.ax.tick_params(labelsize=12.5)
clb.ax.set_title('U', fontsize=12.5)
clb.ax.yaxis.set_major_formatter(FuncFormatter(sci_fmt))
plt.gca().set_aspect('equal', 'box')
plt.gca().set_xlabel("x (m)", fontsize=12.5)
plt.gca().set_ylabel("y (m)", fontsize=12.5)
plt.show()

fig = plt.figure(figsize=(6, 4))
plt.pcolormesh(xc, yc, T_val.reshape(xc.shape), shading='auto', cmap=plt.cm.rainbow)
plt.gca().ticklabel_format(style='scientific', scilimits=(-0.5, 0.5), useMathText=True)
clb = plt.colorbar()
clb.ax.tick_params(labelsize=12.5)
clb.ax.set_title('T', fontsize=12.5)
clb.ax.yaxis.set_major_formatter(FuncFormatter(sci_fmt))
plt.gca().set_aspect('equal', 'box')
plt.gca().set_xlabel("x (m)", fontsize=12.5)
plt.gca().set_ylabel("y (m)", fontsize=12.5)
plt.show()

fig = plt.figure(figsize=(6, 4))
plt.pcolormesh(xc, yc, U_res.reshape(xc.shape), shading='auto', cmap=plt.cm.rainbow)
plt.gca().ticklabel_format(style='scientific', scilimits=(-0.5, 0.5), useMathText=True)
clb = plt.colorbar()
clb.ax.tick_params(labelsize=12.5)
clb.ax.set_title('U_res', fontsize=12.5)
clb.ax.yaxis.set_major_formatter(FuncFormatter(sci_fmt))
plt.gca().set_aspect('equal', 'box')
plt.gca().set_xlabel("x (m)", fontsize=12.5)
plt.gca().set_ylabel("y (m)", fontsize=12.5)
plt.show()

fig = plt.figure(figsize=(6, 4))
plt.pcolormesh(xc, yc, T_res.reshape(xc.shape), shading='auto', cmap=plt.cm.rainbow)
plt.gca().ticklabel_format(style='scientific', scilimits=(-0.5, 0.5), useMathText=True)
clb = plt.colorbar()
clb.ax.tick_params(labelsize=12.5)
clb.ax.set_title('T_res', fontsize=12.5)
clb.ax.yaxis.set_major_formatter(FuncFormatter(sci_fmt))
plt.gca().set_aspect('equal', 'box')
plt.gca().set_xlabel("x (m)", fontsize=12.5)
plt.gca().set_ylabel("y (m)", fontsize=12.5)
plt.show()
