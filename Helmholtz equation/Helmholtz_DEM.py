'''
    Helmholtz equation
    Please refer to Section 3.1.
    If you have any question, please don’t hesitate to contact us (23121084@bjtu.edu.cn).
    '''

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os
import matplotlib.pylab as plt
import time
from matplotlib.ticker import FuncFormatter
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # run the code with the CPU

# tf.set_random_seed(1)

def sci_fmt(x, pos):
    return f'{x:.1e}'

def MLP(x): # In this test, the DEM with sine functions has the best performance.
    with tf.variable_scope('MLP'):
        l1 = tf.layers.dense(x, 32)
        l1 = tf.sin(l1)
        l2 = tf.layers.dense(l1, 32)
        l2 = tf.sin(l2)
        l3 = tf.layers.dense(l2, 32)
        l3 = tf.sin(l3)
        l4 = tf.layers.dense(l3, 32)
        l4 = tf.sin(l4)
        output = tf.layers.dense(l4, 1)
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


def PINN(x, y):

    with tf.GradientTape(persistent=True) as gg:
        gg.watch(x)
        gg.watch(y)
        Ur = MLP(tf.concat([x, y], axis=1))
        Ur = Ur * (x*x-1) * (y*y-1)
    Uxr = gg.gradient(Ur, x)
    Uyr = gg.gradient(Ur, y)
    del gg
    U = Ur
    Ux = Uxr
    Uy = Uyr
    return U, Ux, Uy

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

' Generate coordinate '

xs = -1.0  # Minimum x-coordinate
xl = 1.0   # Maximum x-coordinate
ys = -1.0  # Minimum y-coordinate
yl = 1.0   # Maximum y-coordinate

p1 = 120 # Number of collocation points along the x-direction
p2 = 120 # Number of collocation points along the y-direction

x_list = np.linspace(xs, xl, p1).astype(np.float32)
y_list = np.linspace(ys, yl, p2).astype(np.float32)

long = x_list[1] - x_list[0]

xye, bindexu, bindexb, bindexl, bindexr = coordinat()  # Generate coordinate, where xye denote a matrix of whole coordinates,\
                                                       # bindexu, bindexb, bindexl, bindexr are the marker vectors for the top, bottom, left, and right boundaries, respectively.

x, y = (xye[..., i, tf.newaxis] for i in range(xye.shape[-1]))

x = tf.reshape(tf.constant(x), [p1*p1, 1])
y = tf.reshape(tf.constant(y), [p1*p1, 1])

xy = np.linspace(0.0, xl, p1*p1).astype(np.float32)
xy = tf.reshape(tf.constant(xy), [1, p1*p1])
xy = tf.concat([x, y], 1)

' Obtain physical quantity '

U, Ux, Uy = PINN(x, y)

x = tf.reshape(x, [p1*p1, 1])
y = tf.reshape(y, [p1*p1, 1])

' Loss '

k = 1.0
a1 = 5.0
a2 = 3.0

f = ((a1*np.pi)**2)*tf.sin(a1*np.pi*x)*tf.sin(a2*np.pi*y) + ((a2*np.pi)**2)*tf.sin(a1*np.pi*x)*tf.sin(a2*np.pi*y) - k*k*tf.sin(a1*np.pi*x)*tf.sin(a2*np.pi*y)

l1i = 0.5 * long * long * (Ux*Ux + Uy*Uy) - 0.5 * k * k * U * U * long * long - f * long * long * U
l1 = tf.reduce_mean(tf.reduce_sum(l1i))

l = l1

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 2e-3
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           200, 0.9, staircase=True)
learning_rate = tf.maximum(learning_rate, 1e-6)
optimizer_P = tf.train.AdamOptimizer(learning_rate).minimize(l, global_step=global_step)

' Used for data generation and plotting, unrelated to training '

Ut = tf.sin(a1*np.pi*x) * tf.sin(a2*np.pi*y)  # True value

U_res = tf.abs(U - Ut)  # Absolute error

start = time.perf_counter()

' Training '

with tf.Session() as sess: # The fixed-point training is performed, and the final output is regarded as the prediction.
    sess.run(tf.global_variables_initializer())

    for i in range(15000):
        _, loss_val, lr_val = sess.run([optimizer_P, l, learning_rate])

        if i % 200 == 0:
            l1_val= sess.run(l1)
            print(f"Iter {i}: Loss={loss_val:.2e}, l1={l1_val:.2e}")

    end = time.perf_counter()
    runTime = end - start
    runTime_ms = runTime * 1000
    print("RunTime：", runTime, "seconds")

    U_val, U_truth, U_res_val = sess.run([U, Ut, U_res]) # Due to uniform sampling, the output can be directly regarded as the prediction


xc, yc = np.meshgrid(x_list, y_list)

fig = plt.figure(figsize=(6, 4))
plt.pcolormesh(xc,yc, U_val.reshape(xc.shape), shading='auto', cmap=plt.cm.rainbow)
plt.gca().ticklabel_format(style='scientific',scilimits=(-0.5,0.5),useMathText=True)
clb = plt.colorbar()
clb.ax.tick_params(labelsize=12.5)
clb.ax.set_title('U', fontsize=12.5)
clb.ax.yaxis.set_major_formatter(FuncFormatter(sci_fmt))
plt.gca().set_aspect('equal', 'box')
plt.gca().set_xlabel("x (m)", fontsize=12.5)
plt.gca().set_ylabel("y (m)", fontsize=12.5)
plt.show()

fig = plt.figure(figsize=(6, 4))
plt.pcolormesh(xc, yc, U_truth.reshape(xc.shape), shading='auto', cmap=plt.cm.rainbow)
plt.gca().ticklabel_format(style='scientific',scilimits=(-0.5,0.5),useMathText=True)
clb = plt.colorbar()
clb.ax.tick_params(labelsize=12.5)
clb.ax.set_title('U', fontsize=12.5)
clb.ax.yaxis.set_major_formatter(FuncFormatter(sci_fmt))
plt.gca().set_aspect('equal', 'box')
plt.gca().set_xlabel("x (m)", fontsize=12.5)
plt.gca().set_ylabel("y (m)", fontsize=12.5)
plt.show()

fig = plt.figure(figsize=(6, 4))
plt.pcolormesh(xc, yc, U_res_val.reshape(xc.shape), shading='auto', cmap=plt.cm.rainbow)
plt.gca().ticklabel_format(style='scientific',scilimits=(-0.5,0.5),useMathText=True)
clb = plt.colorbar()
clb.ax.tick_params(labelsize=12.5)
clb.ax.set_title('U', fontsize=12.5)
clb.ax.yaxis.set_major_formatter(FuncFormatter(sci_fmt))
plt.gca().set_aspect('equal', 'box')
plt.gca().set_xlabel("x (m)", fontsize=12.5)
plt.gca().set_ylabel("y (m)", fontsize=12.5)
plt.show()