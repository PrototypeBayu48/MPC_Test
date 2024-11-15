from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, atan, pi
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as Rot

from scipy.integrate import solve_ivp

from NMPC_CasADi_Class_1 import Quad_MPC

from Cont_func_1 import CKBF_STK

import jax

import pickle

def simulate_euler(step_horizon, t0, state_init_quat, u):

    state_init = np.zeros(9)
    state_init[:3] = state_init_quat[:3]
    state_init[3:6] = Rot.from_quat(state_init_quat[3:7]).as_euler('xyz')
    state_init[6:] = state_init_quat[7:]

    u_ctrl = np.squeeze(u[:,1])

    def lin_func(t, x):
        xdot = [x[6], x[7], x[8], u_ctrl[1] ,u_ctrl[2], u_ctrl[3],
                ((cos(x[5])*sin(x[4])*cos(x[3]) + sin(x[5])*sin(x[3]))/mass)*(u_ctrl[0]*0.1*max_thrust),
                ((sin(x[5])*sin(x[4])*cos(x[3]) - cos(x[5])*sin(x[3]))/mass)*(u_ctrl[0]*0.1*max_thrust),
                -gravity + ((cos(x[4])*cos(x[3]))/mass)*(u_ctrl[0]*0.1*max_thrust)]
        return xdot
    soln = solve_ivp(lin_func, [0.0, step_horizon], np.squeeze(state_init), t_eval = [step_horizon])
    xnext = soln.y
    next_state_euler =  np.squeeze(xnext)

    next_state_quat = np.zeros(10)
    next_state_quat[:3] = next_state_euler[:3]
    next_state_quat[3:7] = Rot.from_euler('xyz', next_state_euler[3:6]).as_quat()
    next_state_quat[7:] = next_state_euler[6:]

    return next_state_quat, u_ctrl

def defkbf(n,m):
    return CKBF_STK([n, m, 12+1], 100*np.ones(4), True, jax.nn.swish)

n = 10
m = 4

mass = 2.02
Ixx = 0.011
Iyy = 0.015
Izz = 0.021
gravity = 9.81
l = 0.26
ct = 5.84e-6
cq = 0.06*ct

c1 = (Iyy - Izz)/Ixx
c2 = (Izz - Ixx)/Iyy
c3 = (Iyy - Ixx)/Izz

hover_thrust = mass*gravity
max_thrust = 28.3
min_thrust = 0.0

hover_w2 = (hover_thrust)/(4*ct)
max_w2 = (max_thrust)/(4*ct)
max_tx = ct*l*max_w2
max_ty = ct*l*max_w2
max_tz = 2*cq*max_w2

#dt = 0.02
dt = 0.05

x_train_quat = np.load('x_quat_train_4.npy')
x_train_eul = np.load('x_eul_train_4.npy')
u_train = np.load('u_train_quad_4.npy')

model = 'BLO'

if model == 'BLO':
    koopmat_blo = np.load('matrices_dataset_4_quat_blo_1.npy')
    with open('net_dataset_4_quat_blo_1.pkl','rb') as fp:
        params_blo = pickle.load(fp)

    xscal = np.load('xscal_dataset_4_quat_blo_1.npy')
    uscal = np.load('uscal_dataset_4_quat_blo_1.npy')

    kbf = defkbf(n,m)
    def basis_blo(x):
        return np.array(kbf.encoder(x,params_blo))
    
'''
xtrain = np.load('x_quat_train_3.npy')

for n1 in range(n):
    maxval = np.max(np.abs(xtrain[:,:,n1]))
    if n1 == 2:
        xscal[n1] = 0.1*maxval
    else:
        xscal[n1] = 0.2*maxval
'''
    
nz = len(basis_blo(np.zeros(n)))

nflight = len(x_train_quat[:,0,0])
fltlen = len(x_train_quat[0,:,0])

fltsel = int(np.floor(np.random.rand()*nflight))
startsel = int((np.random.rand()*fltlen)/2)

print(fltsel)
print(startsel)

#testlen = int((5/dt) + 1)
testlen = 25
#testlen = 5

tarray = dt*np.arange(len(x_train_quat[0,:testlen,0]))

x_ref_quat = x_train_quat[fltsel,startsel:startsel+testlen,:]
x_ref_eul = x_train_eul[fltsel,startsel:startsel+testlen,:]
u_ref = u_train[fltsel,startsel:startsel+(testlen-1),:]

uref_norm = (u_ref - uscal[:,1])/uscal[:,0]

zarray = np.zeros((testlen,nz))

xstart_pos = (x_ref_quat[0,:] - xscal[:,1])/xscal[:,0]

zarray[0,:] = np.squeeze(basis_blo(xstart_pos))

for n1 in np.arange(1,testlen):
    kooparray = koopmat_blo[0,:,:] + uref_norm[n1-1,0]*koopmat_blo[1,:,:] + uref_norm[n1-1,1]*koopmat_blo[2,:,:] +  + uref_norm[n1-1,2]*koopmat_blo[3,:,:] + uref_norm[n1-1,3]*koopmat_blo[4,:,:]
    zarray[n1,:] = zarray[n1-1,:] + dt*kooparray@zarray[n1-1,:]

x_koop_quat = zarray[:,1:n+1]

x_koop_quat = (x_koop_quat*xscal[:,0]) + xscal[:,1]

x_koop_eul = np.zeros((testlen,n-1))
x_koop_eul[:,:3] = x_koop_quat[:testlen,:3]
x_koop_eul[:,6:] = x_koop_quat[:testlen,7:]

for n1 in range(testlen):
    x_koop_eul[n1,3:6] = Rot.from_quat(x_koop_quat[n1,3:7]).as_euler('xyz')

plt.figure()
plt.plot(tarray, x_ref_eul[:,0], 'c-', label='Reference')
plt.plot(tarray, x_koop_eul[:,0], 'r--', label='Koopman')
plt.xlabel('Time (s)')
plt.ylabel('x (m)')
plt.legend()
plt.show()

plt.figure()
plt.plot(tarray, x_ref_eul[:,1], 'c-', label='Reference')
plt.plot(tarray, x_koop_eul[:,1], 'r--', label='Koopman')
plt.xlabel('Time (s)')
plt.ylabel('y (m)')
plt.legend()
plt.show()

plt.figure()
plt.plot(tarray, x_ref_eul[:,2], 'c-', label='Reference')
plt.plot(tarray, x_koop_eul[:,2], 'r--', label='Koopman')
plt.xlabel('Time (s)')
plt.ylabel('z (m)')
plt.legend()
plt.show()

plt.figure()
plt.plot(tarray, (x_ref_eul[:,3]/np.pi)*180, label='Reference phi')
plt.plot(tarray, (x_ref_eul[:,4]/np.pi)*180, label='Reference theta')
plt.plot(tarray, (x_ref_eul[:,5]/np.pi)*180, label='Reference psi')
plt.plot(tarray, (x_koop_eul[:,3]/np.pi)*180, '--', label='Koopman phi')
plt.plot(tarray, (x_koop_eul[:,4]/np.pi)*180, '--', label='Koopman theta')
plt.plot(tarray, (x_koop_eul[:,5]/np.pi)*180, '--', label='Koopman psi')
plt.xlabel('Time (s)')
plt.ylabel('Angle (deg)')
plt.legend()
plt.show()

plt.figure()
plt.plot(tarray, x_ref_quat[:,3], label='Reference qx')
plt.plot(tarray, x_ref_quat[:,4], label='Reference qy')
plt.plot(tarray, x_ref_quat[:,5], label='Reference qz')
plt.plot(tarray, x_ref_quat[:,6], label='Reference qw')
plt.plot(tarray, x_koop_quat[:,3], '--', label='Koopman qx')
plt.plot(tarray, x_koop_quat[:,4], '--', label='Koopman qy')
plt.plot(tarray, x_koop_quat[:,5], '--', label='Koopman qz')
plt.plot(tarray, x_koop_quat[:,6], '--', label='Koopman qw')
plt.xlabel('Time (s)')
plt.ylabel('Quaternion')
plt.legend()
plt.show()

plt.figure()
plt.stairs(u_ref[:,0]*0.1*max_thrust, tarray, label='Thrust')
plt.plot([tarray[0],tarray[-1]], hover_thrust*np.ones(2), '--', label='Hover Thrust')
plt.plot([tarray[0],tarray[-1]], max_thrust*np.ones(2), '--', label='Max Thrust')
plt.xlabel('Time (s)')
plt.ylabel('Thrust (N)')
plt.legend()
plt.show()

plt.figure()
plt.stairs(u_ref[:,1], tarray, label='omega_x')
plt.stairs(u_ref[:,2], tarray, label='omega_y')
plt.stairs(u_ref[:,3], tarray, label='omega_z')
plt.xlabel('Time (s)')
plt.ylabel('Rotation Rate (rad/s)')
plt.legend()
plt.show()

plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(x_ref_eul[:,0], x_ref_eul[:,1], x_ref_eul[:,2], 'c-', label='Reference Trajectory')
ax.plot3D(x_koop_eul[:,0], x_koop_eul[:,1], x_koop_eul[:,2], 'r--', label='Koopman Trajectory')
plt.legend()
ax.set_xlabel('x, m')
ax.set_ylabel('y, m')
ax.set_zlabel('z, m')
plt.show()



