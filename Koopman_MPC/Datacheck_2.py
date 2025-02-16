from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, atan, pi
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as Rot

from scipy.integrate import solve_ivp

from NMPC_CasADi_Class_1 import Quad_MPC

import os

quadspec_crazyflie = [0.0336, 1.7e-5, 1.7e-5, 2.9e-5, 0.04, 3.1582e-10, 7.9379e-12, 0.057*9.81]
quadspec_iris = [2.02, 0.011, 0.015, 0.021, 0.26, 5.84e-6, 0.06*5.84e-6, 28.3]

quadspec = quadspec_crazyflie

n = 10
m = 4

mass = quadspec[0]
Ixx = quadspec[1]
Iyy = quadspec[2]
Izz = quadspec[3]
gravity = 9.81
l = quadspec[4]
ct = quadspec[5]
cq = quadspec[6]

c1 = (Iyy - Izz)/Ixx
c2 = (Izz - Ixx)/Iyy
c3 = (Iyy - Ixx)/Izz

hover_thrust = mass*gravity
max_thrust = quadspec[7]
min_thrust = 0.0

hover_w2 = (hover_thrust)/(4*ct)
max_w2 = (max_thrust)/(4*ct)
max_tx = ct*l*max_w2
max_ty = ct*l*max_w2
max_tz = 2*cq*max_w2

dt = 0.02

xarray_quat = np.load(os.getcwd() + '/traindata/crazyflie_x_quat_train_3.npy')
xarray_eul = np.load(os.getcwd() + '/traindata/crazyflie_x_eul_train_3.npy')
uarray = np.load(os.getcwd() + '/traindata/crazyflie_u_train_quad_3.npy')

'''
xarray_quat = np.load(os.getcwd() + '/traindata/x_quat_train_6.npy')
xarray_eul = np.load(os.getcwd() + '/traindata/x_eul_train_6.npy')
uarray = np.load(os.getcwd() + '/traindata/u_train_quad_6.npy')
'''

xarray_total = xarray_quat[0,:,:]
xarray_euler_total = xarray_eul[0,:,:]
uarray_total = uarray[0,:,:]

tarray = dt*np.arange(len(xarray_euler_total[:,0]))

thrust_total = ct*np.sum(uarray_total, axis=1)

fig1, ax1 = plt.subplots(3,sharex=True)
for n1 in np.arange(3):
    if n1 == 0:
        ax1[n1].plot(tarray,xarray_euler_total[:,n1],'b-')
    else:
        ax1[n1].plot(tarray,xarray_euler_total[:,n1],'b-')
ax1[0].set(ylabel='x (m)')
ax1[1].set(ylabel='y (m)')
ax1[2].set(ylabel='z (m)')
fig1.supxlabel('Time (s)')

plt.show()
plt.figure()
plt.plot(tarray, (xarray_euler_total[:,3]/np.pi)*180, label='phi')
plt.plot(tarray, (xarray_euler_total[:,4]/np.pi)*180, label='theta')
plt.plot(tarray, (xarray_euler_total[:,5]/np.pi)*180, label='psi')
plt.xlabel('Time (s)')
plt.ylabel('Angle (deg)')
plt.legend()
plt.show()

plt.figure()
plt.plot(tarray, xarray_total[:,3], label='qx')
plt.plot(tarray, xarray_total[:,4], label='qy')
plt.plot(tarray, xarray_total[:,5], label='qz')
plt.plot(tarray, xarray_total[:,6], label='qw')
plt.xlabel('Time (s)')
plt.ylabel('Quaternion')
plt.legend()
plt.show()

plt.figure()
#plt.stairs(uarray_total[:,0]*0.1*max_thrust, tarray, label='Thrust')
plt.stairs(uarray_total[:,0], tarray, label='Thrust')
plt.plot([tarray[0],tarray[-1]], hover_thrust*np.ones(2), '--', label='Hover Thrust')
plt.plot([tarray[0],tarray[-1]], max_thrust*np.ones(2), '--', label='Max Thrust')
plt.xlabel('Time (s)')
plt.ylabel('Thrust (N)')
plt.legend()
plt.show()

plt.figure()
plt.stairs(uarray_total[:,1], tarray, label='omega_x')
plt.stairs(uarray_total[:,2], tarray, label='omega_y')
plt.stairs(uarray_total[:,3], tarray, label='omega_z')
plt.xlabel('Time (s)')
plt.ylabel('Rotation Rate (rad/s)')
plt.legend()
plt.show()

plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(xarray_euler_total[:,0], xarray_euler_total[:,1], xarray_euler_total[:,2], 'c-', label='Trajectory')
plt.legend()
ax.set_xlabel('x, m')
ax.set_ylabel('y, m')
ax.set_zlabel('z, m')
plt.show()



