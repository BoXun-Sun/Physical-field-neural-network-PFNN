'''
    Heat equation
    Please refer to Section 3.2.
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

def coordinat():
    xye = []
    lflag = []
    rflag = []
    bflag = []
    uflag = []
    for j in range(0, p2):
        for i in range(0, p1):
            xye.append([x_list[i], t_list[j]])
            if i == 0:
                lflag.append([x_list[i], t_list[j]])
            if i == p1 - 1:
                rflag.append([x_list[i], t_list[j]])
            if j == 0:
                bflag.append([x_list[i], t_list[j]])
            if j == p2 - 1:
                uflag.append([x_list[i], t_list[j]])
    xye = np.array(xye).reshape(p1*p2, 2)
    bindexu = (np.where((xye[:, None] == uflag).all(-1))[0]).reshape(p1, 1)
    bindexb = (np.where((xye[:, None] == bflag).all(-1))[0]).reshape(p1, 1)
    bindexl = (np.where((xye[:, None] == lflag).all(-1))[0]).reshape(p2, 1)
    bindexr = (np.where((xye[:, None] == rflag).all(-1))[0]).reshape(p2, 1)
    return xye, bindexu, bindexb, bindexl, bindexr

def FDFDs1(U, choose):
    if choose == 0:
        # First-order derivative with respect to t
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

def MLP(x):

    with tf.variable_scope('MLP'):
        l1 = tf.layers.dense(x, 32)
        l1 = tf.nn.tanh(l1)
        l2 = tf.layers.dense(l1, 32)
        l2 = tf.nn.tanh(l2)
        l3 = tf.layers.dense(l2, 32)
        l3 = tf.nn.tanh(l3)
        l4 = tf.layers.dense(l3, 32)
        l4 = tf.nn.tanh(l4)
        l5 = tf.layers.dense(l4, 32)
        l5 = tf.nn.tanh(l5)
        output = tf.layers.dense(l5, 1)
    return output

def PINN(x, t):
    with tf.GradientTape(persistent=True) as g:
        g.watch(x)
        g.watch(t)
        with tf.GradientTape(persistent=True) as gg:
            gg.watch(x)
            gg.watch(t)
            P = tf.sin(50 * np.pi * x)
            D = x * (1 - x) * t
            N = MLP(tf.concat([x, t], axis=1))
            U = P + D * N
        Ux = gg.gradient(U, x)
        Ut = gg.gradient(U, t)
        del gg
    Uxx = g.gradient(Ux, x)
    del g
    return U, Ux, Ut, Uxx

def H1(Ut, U):

    Ue = U - Ut

    Ue = tf.reshape(Ue, [p2, p1])

    Ux = FDFDs1(tf.reshape(Ue, [p2, p1]), 1)
    Uy = FDFDs1(tf.reshape(Ue, [p2, p1]), 0)


    H1 = tf.sqrt(tf.reduce_sum((Ux * Ux + Uy * Uy) * long * long))

    return H1

def MAE(Ut, U):

    U_res = U - Ut

    return tf.reduce_mean(tf.abs(U_res)), U_res, tf.abs(U_res)


def L2(Ut, U):

    L_2 = (U - Ut)

    return tf.sqrt(tf.reduce_sum((tf.square(L_2)) * long  * long))

' Generate coordinate '

xs = 0.0  # Minimum x-coordinate
xl = 1.0   # Maximum x-coordinate
ts = 0.0  # Minimum t-coordinate
tl = 1.0   # Maximum t-coordinate

p1 = 250  # Number of collocation points along the x-direction
p2 = 250  # Number of collocation points along the t-direction

x_list = np.linspace(xs,xl,p1).astype(np.float32)
t_list = np.linspace(ts,tl,p2).astype(np.float32)

long = x_list[1] - x_list[0]

xye, bindexu, bindexb, bindexl, bindexr = coordinat()  # Generate coordinate, where xte denote a matrix of whole coordinates,\
                                                       # bindexu, bindexb, bindexl, bindexr are the marker vectors for the top, bottom, left, and right boundaries, respectively.

x, t = (xye[..., i, tf.newaxis] for i in range(xye.shape[-1]))

x = tf.reshape(tf.constant(x), [p1*p2, 1])
t = tf.reshape(tf.constant(t), [p1*p2, 1])

' Obtain physical quantity '

U, Ux, Ut, Uxx = PINN(x, t)

' Loss '

k = 1/(50*np.pi)/(50*np.pi)

l11 = Ut - k * Uxx
l1 = tf.reduce_mean(tf.square(l11))

l = l1

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 1e-3
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           200, 0.90, staircase=True) # According to extensive tests
learning_rate = tf.maximum(learning_rate, 2e-5)
optimizer_P = tf.train.AdamOptimizer(learning_rate).minimize(l, global_step=global_step)

' Used for data generation and plotting, unrelated to training '

Ut = tf.exp(-t) * tf.sin(50*np.pi*x)

Ut = tf.reshape(Ut, [p1 * p2, 1])

U_res = tf.abs(U-Ut)

start = time.perf_counter()

' Train model '

with tf.Session() as sess: # The fixed-point training is performed, and the final output is regarded as the prediction.
    sess.run(tf.global_variables_initializer())

    for i in range(0, 5000):
        _, l_o = sess.run([optimizer_P, l])

        if i % 200 == 0:
            print(i,l_o)

    print(i, l_o) # Due to uniform sampling, the output can be directly regarded as the prediction
    end = time.perf_counter()
    runTime = end - start
    runTime_ms = runTime * 1000
    print("RunTime：", runTime, "seconds")

    U_val, U_truth, U_res_val = sess.run([U, Ut, U_res])

xc, tc = np.meshgrid(x_list, t_list)

fig = plt.figure(figsize=(6, 4))
plt.pcolormesh(xc, tc, U_val.reshape(xc.shape), shading='auto', cmap=plt.cm.rainbow)
plt.gca().ticklabel_format(style='scientific', scilimits=(-0.5, 0.5), useMathText=True)
clb = plt.colorbar()
clb.ax.tick_params(labelsize=12.5)
clb.ax.set_title('U', fontsize=12.5)
clb.ax.yaxis.set_major_formatter(FuncFormatter(sci_fmt))
plt.gca().set_aspect('equal', 'box')
plt.gca().set_xlabel("x", fontsize=12.5)
plt.gca().set_ylabel("t", fontsize=12.5)
plt.show()

fig = plt.figure(figsize=(6, 4))
plt.pcolormesh(xc, tc, U_truth.reshape(xc.shape), shading='auto', cmap=plt.cm.rainbow)
plt.gca().ticklabel_format(style='scientific', scilimits=(-0.5,0.5),useMathText=True)
clb = plt.colorbar()
clb.ax.tick_params(labelsize=12.5)
clb.ax.set_title('U', fontsize=12.5)
clb.ax.yaxis.set_major_formatter(FuncFormatter(sci_fmt))
plt.gca().set_aspect('equal', 'box')
plt.gca().set_xlabel("x", fontsize=12.5)
plt.gca().set_ylabel("t", fontsize=12.5)
plt.show()

fig = plt.figure(figsize=(6, 4))
plt.pcolormesh(xc, tc, U_res_val.reshape(xc.shape), shading='auto', cmap=plt.cm.rainbow)
plt.gca().ticklabel_format(style='scientific', scilimits=(-0.5,0.5),useMathText=True)
clb = plt.colorbar()
clb.ax.tick_params(labelsize=12.5)
clb.ax.set_title('U', fontsize=12.5)
clb.ax.yaxis.set_major_formatter(FuncFormatter(sci_fmt))
plt.gca().set_aspect('equal', 'box')
plt.gca().set_xlabel("x", fontsize=12.5)
plt.gca().set_ylabel("t", fontsize=12.5)
plt.show()