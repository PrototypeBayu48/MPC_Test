from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, atan, pi
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as Rot

from scipy.integrate import solve_ivp

from NMPC_CasADi_Class_1 import Quad_MPC

quadspec_crazyflie = [0.0336, 1.7e-5, 1.7e-5, 2.9e-5, 0.04, 3.1582e-10, 7.9379e-12, 0.057*9.81]
quadspec_iris = [2.02, 0.011, 0.015, 0.021, 0.26, 5.84e-6, 0.06*5.84e-6, 28.3]

quadspec = quadspec_iris

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

#dt = 0.02
dt = 0.05

xarray_quat = np.load('x_quat_train_5.npy')
xarray_eul = np.load('x_eul_train_5.npy')
uarray = np.load('u_train_quad_5.npy')

fltlen = len(xarray_eul[:,0,0])

trajlen = len(xarray_eul[0,:,0])

xarray_quat_mag = np.zeros([fltlen,trajlen])

xarray_rotvec = np.zeros([fltlen,trajlen,n])

xarray_rotvec_mag = np.zeros([fltlen,trajlen])

for n1 in range(fltlen):
    for n2 in range(trajlen):
        xarray_rotvec[n1,n2,3:6] = Rot.from_euler('xyz', xarray_eul[n1,n2,3:6]).as_rotvec()
        xarray_rotvec_mag[n1,n2] = np.linalg.norm(xarray_rotvec[n1,n2,3:6])
        xarray_quat_mag[n1,n2] = np.linalg.norm(xarray_quat[n1,n2,3:7])

tarray = dt*np.arange(trajlen)

plt.figure()
plt.plot(tarray, xarray_quat_mag[0,:])
plt.xlabel('Time (s)')
plt.ylabel('Magnitude')
plt.show()

fig1, ax1 = plt.subplots(4,sharex=True)

ax1[0].plot(tarray, xarray_rotvec[0,:,3], label='e_x')
ax1[1].plot(tarray, xarray_rotvec[0,:,4], label='e_y')
ax1[2].plot(tarray, xarray_rotvec[0,:,5], label='e_z')
ax1[3].plot(tarray, (xarray_rotvec_mag[0,:]/pi)*180, label='Magnitude')

ax1[0].set(ylabel='e_x')
ax1[1].set(ylabel='e_y')
ax1[2].set(ylabel='e_z')
ax1[3].set(ylabel='Magnitude (deg)')
fig1.legend()
fig1.supxlabel('Time (s)')
plt.show()



