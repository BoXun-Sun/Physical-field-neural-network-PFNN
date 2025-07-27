import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from sympy import *
import concurrent.futures
import xlwt
import os
import matplotlib.pylab as plt
import time
from matplotlib.ticker import FuncFormatter
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # run the code with the CPU

tf.set_random_seed(10)

def sci_fmt(x, pos):
    return f'{x:.1e}'

def MLP1(x):
    with tf.variable_scope('MLP1'):
        l1 = tf.layers.dense(x, 64)
        l1 = tf.nn.tanh(l1)
        l2 = tf.layers.dense(l1, 64)
        l2 = tf.nn.tanh(l2)
        l3 = tf.layers.dense(l2, 64)
        l3 = tf.nn.tanh(l3)
        l4 = tf.layers.dense(l3, 64)
        l4 = tf.nn.tanh(l4)
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
    with tf.GradientTape(persistent=True) as g:
        g.watch(x)
        g.watch(y)
        with tf.GradientTape(persistent=True) as gg:
            gg.watch(x)
            gg.watch(y)
            Ur = MLP1(tf.concat([x, y], axis=1))
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

def get_mlp1_weights():
    # Obtain trainable parameters from MLP1
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='MLP1')
    # Separate weights and biases
    weights = [var for var in trainable_vars if 'kernel' in var.name]
    biases = [var for var in trainable_vars if 'bias' in var.name]
    return weights, biases

' Generate coordinate '

xs = -1.0  # Minimum x-coordinate
xl = 1.0   # Maximum x-coordinate
ys = -1.0  # Minimum y-coordinate
yl = 1.0   # Maximum y-coordinate

p1 = 100 # Number of collocation points along the x-direction
p2 = 100 # Number of collocation points along the y-direction

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

U, Ux, Uxx, Uy, Uyy = PINN(x, y)

x = tf.reshape(x, [p1*p1, 1])
y = tf.reshape(y, [p1*p1, 1])

' Loss function, please refer to Eq.(7) '

k = 1.0  # Please refer to Eq.(7)
a1 = 5.0
a2 = 3.0

l1i = Uxx + Uyy + k**2 * U + ((a1*np.pi)**2)*tf.sin(a1*np.pi*x)*tf.sin(a2*np.pi*y) + ((a2*np.pi)**2)*tf.sin(a1*np.pi*x)*tf.sin(a2*np.pi*y) - k*k*tf.sin(a1*np.pi*x)*tf.sin(a2*np.pi*y)
l1 = tf.reduce_mean(tf.square(l1i))

l2i = tf.gather_nd(U, bindexb) - 0.0
l2 = tf.reduce_mean(tf.square(l2i))

l3i = tf.gather_nd(U, bindexu) - 0.0
l3 = tf.reduce_mean(tf.square(l3i))

l4i = tf.gather_nd(U, bindexl) - 0.0
l4 = tf.reduce_mean(tf.square(l4i))

l5i = tf.gather_nd(U, bindexr) - 0.0
l5 = tf.reduce_mean(tf.square(l5i))

weights, biases = get_mlp1_weights()

' Gradient-based learning rate annealing, please refer to Appendix A '

ad = tf.Variable(1.0, dtype=tf.float32, trainable=False)  # Weight of boundary loss
ad_const = tf.stop_gradient(ad)
lb =  (l2 + l3 + l4 + l5)
l = l1 + lb

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 1e-3
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           400, 0.9, staircase=True)
optimizer_P = tf.train.AdamOptimizer(learning_rate).minimize(l, global_step=global_step)

trainable_vars = tf.trainable_variables()

g_re = tf.gradients(l1, trainable_vars) # Compute the gradient of the loss function of governing equation with respect to weights of MLP1

g_be = tf.gradients(lb, trainable_vars) # Compute the gradient of loss functions of boundary conditions with respect to weights of MLP1

max_grads_re = [tf.reduce_max(tf.abs(g)) for g in g_re]
global_max_g = tf.reduce_max(tf.stack(max_grads_re))
mean_grads_be = [tf.reduce_mean(tf.abs(g)) for g in g_be]
global_mean_g = tf.reduce_mean(tf.stack(mean_grads_be))

ad_new = global_max_g / global_mean_g

beta = 0.9  # The annealing coefficient is set to be 0.9
update_ad_op = tf.assign(ad, (1.0 - beta) * ad_new + beta * ad)

Ut = tf.sin(a1*np.pi*x) * tf.sin(a2*np.pi*y)

U_res = U - Ut  # Absolute error

start = time.perf_counter()

# ---------------------------- 训练循环 ----------------------------
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        _, loss_val, lr_val, _ = sess.run([optimizer_P, l, learning_rate, update_ad_op])

        if i % 200 == 0:
            ad_val, l1_val, lb_val = sess.run([ad, l1, lb])
            print(f"Iter {i}: Loss={loss_val:.2e}, l1={l1_val:.2e}, lb={lb_val:.2e}, lr={lr_val:.1e}, ad={ad_val:.1f}")
    u1, u2, u3 = sess.run([U, Ut, U_res])
end = time.perf_counter()
runTime = end - start
runTime_ms = runTime * 1000
print("运行时间：", runTime, "秒")
print("运行时间：", runTime_ms, "毫秒")

xc, yc = np.meshgrid(x_list, y_list)

fig = plt.figure(figsize=(6, 4))
plt.pcolormesh(xc,yc, u1.reshape(xc.shape), shading='auto', cmap=plt.cm.rainbow)
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
plt.pcolormesh(xc, yc, u2.reshape(xc.shape), shading='auto', cmap=plt.cm.rainbow)
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
plt.pcolormesh(xc, yc, u3.reshape(xc.shape), shading='auto', cmap=plt.cm.rainbow)
plt.gca().ticklabel_format(style='scientific',scilimits=(-0.5,0.5),useMathText=True)
clb = plt.colorbar()
clb.ax.tick_params(labelsize=12.5)
clb.ax.set_title('U', fontsize=12.5)
clb.ax.yaxis.set_major_formatter(FuncFormatter(sci_fmt))
plt.gca().set_aspect('equal', 'box')
plt.gca().set_xlabel("x (m)", fontsize=12.5)
plt.gca().set_ylabel("y (m)", fontsize=12.5)
plt.show()

book = xlwt.Workbook()
sheet = book.add_sheet('sheet1', cell_overwrite_ok=True)
for i in range(len(u1)):
    sheet.write(i, 0, float(u1[i]))
    sheet.write(i, 1, float(u2[i]))
    sheet.write(i, 2, float(u3[i]))
book.save('2D_Helmholtz_PINN.xls')