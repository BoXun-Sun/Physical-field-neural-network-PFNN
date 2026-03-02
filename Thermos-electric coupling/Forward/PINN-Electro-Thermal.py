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

# tf.set_random_seed(1)

def sci_fmt(x, pos):
    return f'{x:.1e}'

def MLP1(x): # Please refer to Section 3.4
    with tf.variable_scope('MLP1'):
        l1 = tf.layers.dense(x, 32)
        l1 = tf.nn.tanh(l1)
        l2 = tf.layers.dense(l1, 32)
        l2 = tf.nn.tanh(l2)
        output = tf.layers.dense(l2, 1)
    return output

def MLP2(x): # Please refer to Section 3.4
    with tf.variable_scope('MLP2'):
        l1 = tf.layers.dense(x, 32)
        l1 = tf.nn.tanh(l1)
        l2 = tf.layers.dense(l1, 32)
        l2 = tf.nn.tanh(l2)
        output = tf.layers.dense(l2, 1)
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


def PINN1(x, y):
    with tf.GradientTape(persistent=True) as g:
        g.watch(x)
        g.watch(y)
        with tf.GradientTape(persistent=True) as gg:
            gg.watch(x)
            gg.watch(y)
            Ur = MLP1(tf.concat([x, y], axis=1)) * (0.5*0.5 - y * y) + (0.5 + y)
        Uxr = gg.gradient(Ur, x)
        Uyr = gg.gradient(Ur, y)
        del gg
    Uxxr = g.gradient(Uxr, x)
    Uyyr = g.gradient(Uyr, y)
    U = Ur
    Ux = Uxr
    Uy = Uyr
    Uxx = Uxxr
    Uyy = Uyyr
    del g
    return U, Ux, Uy, Uxx, Uyy

def PINN2(x, y):
    with tf.GradientTape(persistent=True) as g:
        g.watch(x)
        g.watch(y)
        with tf.GradientTape(persistent=True) as gg:
            gg.watch(x)
            gg.watch(y)
            Ur = MLP2(tf.concat([x, y], axis=1)) * (x*x - 0.5*0.5) * (y*y - 0.5*0.5) + 273.0
        Uxr = gg.gradient(Ur, x)
        Uyr = gg.gradient(Ur, y)
        del gg
    Uxxr = g.gradient(Uxr, x)
    Uyyr = g.gradient(Uyr, y)
    U = Ur
    Uxx = Uxxr
    Uyy = Uyyr
    del g
    return U, Uxx, Uyy

def import_data():

    workbook = xlrd.open_workbook('electro-thermal-data.xls')
    sheet = workbook.sheet_by_index(0)  # 获取第一个工作表

    first_column = []
    second_column = []

    for row_idx in range(sheet.nrows):

        first_column.append(sheet.cell_value(row_idx, 3))
        second_column.append(sheet.cell_value(row_idx, 4))

    return first_column, second_column

U_truth, T_truth = import_data()

U_truth = np.array(U_truth, dtype=np.float32).reshape(-1, 1)
T_truth = np.array(T_truth, dtype=np.float32).reshape(-1, 1)

' Generate coordinate '

xs = -0.5  # Minimum x-coordinate
xl = 0.5   # Maximum x-coordinate
ys = -0.5  # Minimum y-coordinate
yl = 0.5  # Maximum y-coordinate

p1 = 50  # Number of collocation points along the x-direction
p2 = 50  # Number of collocation points along the y-direction

x_list = np.linspace(xs, xl, p1).astype(np.float32)
y_list = np.linspace(ys, yl, p2).astype(np.float32)

xye, bindexu, bindexb, bindexl, bindexr = coordinat()  # Generate coordinate, where xye denote a matrix of whole coordinates,\
                                                       # bindexu, bindexb, bindexl, bindexr are the marker vectors for the top, bottom, left, and right boundaries, respectively.

x, y = (xye[..., i, tf.newaxis] for i in range(xye.shape[-1]))

x = tf.reshape(tf.constant(x), [p1*p1, 1])
y = tf.reshape(tf.constant(y), [p1*p1, 1])

' Obtain physical quantity '

U, Ux, Uy, Uxx, Uyy = PINN1(x, y)
T, Txx, Tyy = PINN2(x, y)

' Loss '

sigma = 1.0
k = 1/500

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

l = lg1 + lg2 + lb1 + lb2

' Optimization '

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 1e-4
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           200, 0.9, staircase=True)
learning_rate = tf.maximum(learning_rate, 2e-5)
optimizer_P = tf.train.AdamOptimizer(learning_rate).minimize(l, global_step=global_step)

start = time.perf_counter()

with tf.Session() as sess:# The fixed-point training is performed, and the final output is regarded as the prediction.
    sess.run(tf.global_variables_initializer())

    for i in range(20000):
        _, loss_val, lr_val = sess.run([optimizer_P, l, learning_rate])

        if i % 200 == 0:
            lg1_val, lg2_val, lb1_val, lb2_val = sess.run([lg1, lg2, lb1, lb2])
            print(f"Iter {i}: Loss={loss_val:.2e}, lg1={lg1_val:.2e}, lg2={lg2_val:.2e}, lb1={lb1_val:.2e}, lb2={lb2_val:.2e}")

    print(f"Iter {i}: Loss={loss_val:.2e}, lg1={lg1_val:.2e}, lg2={lg2_val:.2e}, lb1={lb1_val:.2e}, lb2={lb2_val:.2e}") # Due to uniform sampling, the output can be directly regarded as the prediction

    end = time.perf_counter()
    runTime = end - start
    print("RunTime：", runTime, "seconds")

    U_val, T_val = sess.run([U, T])

U_res = abs(U_val - U_truth)
T_res = abs(T_val - T_truth)

MAE = sum(T_res) / len(T_res)

xc, yc = np.meshgrid(x_list, y_list)

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