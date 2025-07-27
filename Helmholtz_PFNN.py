import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import xlwt
import os
import matplotlib.pylab as plt
import time
from matplotlib.ticker import FuncFormatter
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # run the code with the CPU

tf.set_random_seed(1)

def sci_fmt(x, pos):
    return f'{x:.1e}'

def MLP1(x): # Please refer to Section 2.2
    with tf.variable_scope('MLP1'):
        l1 = tf.layers.dense(x, 64)
        l1 = tf.nn.relu(l1)
        l2 = tf.layers.dense(l1, 64)
        l2 = tf.nn.relu(l2)
        l3 = tf.layers.dense(l2, 64)
        l3 = tf.nn.relu(l3)
        l4 = tf.layers.dense(l3, 64)
        l4 = tf.nn.relu(l4)
        output = tf.layers.dense(l4, p1*p1)
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
    # Second-order derivative with respect to y
    if choose == 0:
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
    Ur = MLP1(xy)   # Output physical field
    Ur = tf.reshape(Ur, [p2 * p1, 1])
    U = Ur
    U = tf.reshape(U, [p2, p1])
    Ux = FDFDs1(U, 1)   # First-order derivative with respect to x
    Uy = FDFDs1(U, 0)   # First-order derivative with respect to y
    Uxx = FDFDs2(U, 1)  # Second-order derivative with respect to x
    Uyy = FDFDs2(U, 0)  # Second-order derivative with respect to y
    U = tf.reshape(U, [p1*p2, 1])
    return U, Ux, Uxx, Uy, Uyy, Ur


def get_weights():
    # Obtain trainable parameters from MLP1
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='MLP1')
    # Separate weights and biases
    weights = [var for var in trainable_vars if 'kernel' in var.name]
    biases = [var for var in trainable_vars if 'bias' in var.name]
    return weights, biases


' Generate coordinate '

xs = -1.0 # Minimum x-coordinate
xl = 1.0  # Maximum x-coordinate
ys = -1.0 # Minimum y-coordinate
yl = 1.0  # Maximum y-coordinate

p1 = 100 # Number of collocation points along the x-direction
p2 = 100 # Number of collocation points along the y-direction

x_list = np.linspace(xs, xl, p1).astype(np.float32)
y_list = np.linspace(ys, yl, p2).astype(np.float32)

long = x_list[1] - x_list[0] # Step size

xye, bindexu, bindexb, bindexl, bindexr = coordinat() # Generate coordinate, where xye denote a matrix of whole coordinates,\
                                                      # bindexu, bindexb, bindexl, bindexr are the marker vectors for the top, bottom, left, and right boundaries, respectively.

x, y = (xye[..., i, tf.newaxis] for i in range(xye.shape[-1]))

x = tf.reshape(tf.constant(x), [1, p1*p1])
y = tf.reshape(tf.constant(y), [1, p1*p1])

' Encoding '

v1 = tf.Variable(tf.ones_like(x, dtype=tf.float32)) # Encoding coefficient matrix for x
v2 = tf.Variable(tf.ones_like(y, dtype=tf.float32)) # Encoding coefficient matrix for y

xy = x * v1 + y * v2 # Encoding
xy = tf.concat(xy, 1)

' Obtain physical quantity '

U, Ux, Uxx, Uy, Uyy, Ur = PINN1()

x = tf.reshape(x, [p1*p1, 1])
y = tf.reshape(y, [p1*p1, 1])

k = 1.0
a1 = 5.0
a2 = 3.0

' Loss function, please refer to Eq.(7) '

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

' Gradient-based learning rate annealing, please refer to Appendix A '

ad = tf.Variable(1.0, dtype=tf.float32, trainable=False) # Weight of boundary loss
ad_const = tf.stop_gradient(ad)
lb = ad_const * (l2 + l3 + l4 + l5)

l = l1 + lb

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 1e-3
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           400, 0.9, staircase=True)  # The annealing coefficient is set to be 0.9
optimizer_P = tf.train.AdamOptimizer(learning_rate).minimize(l, global_step=global_step)

trainable_vars, _ = get_weights()

g_re = tf.gradients(l1, trainable_vars) # Compute the gradient of the loss function of governing equation with respect to weights of MLP1

g_be = tf.gradients(lb, trainable_vars) # Compute the gradient of loss functions of boundary conditions with respect to weights of MLP1

max_grads_re = [tf.reduce_max(tf.abs(g)) for g in g_re]
global_max_g = tf.reduce_max(tf.stack(max_grads_re))
mean_grads_be = [tf.reduce_mean(tf.abs(g)) for g in g_be]
global_mean_g = tf.reduce_mean(tf.stack(mean_grads_be))

ad_new = global_max_g / global_mean_g

beta = 0.9
update_ad_op = tf.assign(ad, (1.0 - beta) * ad_new + beta * ad)

' Used for data generation and plotting, unrelated to training '

Ut = tf.sin(a1*np.pi*x) * tf.sin(a2*np.pi*y) # True value

U_res = tf.abs(U - Ut) # Absolute error

start = time.perf_counter()

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
print("RunTime：", runTime, "seconds")

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
book.save('Helmholtz.xls')