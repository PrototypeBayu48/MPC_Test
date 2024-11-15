from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, pi
import matplotlib.pyplot as plt
from simulation_code import simulate

from scipy.spatial.transform import Rotation as Rot

from scipy.integrate import solve_ivp

from Quad_MPC_CasADi_Class_2 import Quad_MPC

import random

def quat2eul(x_quat):
    x_euler = np.zeros(9)
    x_euler[:3] = x_quat[:3]
    x_euler[3:6] = Rot.from_quat(x_quat[3:7]).as_euler('xyz')
    x_euler[6:] = x_quat[7:]

    return x_euler

def eul2quat(x_euler):
    x_quat = np.zeros(10)
    x_quat[:3] = x_euler[:3]
    x_quat[3:7] = Rot.from_euler('xyz', x_euler[3:6]).as_quat()
    x_quat[7:] = x_euler[6:]
    
    return x_quat

def simulate_quat(step_horizon, t0, state_init, u):

    u_ctrl = np.squeeze(u)

    def lin_func(t, x):
        xdot = [x[7], x[8], x[9],
                0.5*(u_ctrl[1]*x[6] - u_ctrl[2]*x[5] + u_ctrl[3]*x[4]),
                0.5*(u_ctrl[1]*x[5] + u_ctrl[2]*x[6] - u_ctrl[3]*x[3]),
                0.5*(-u_ctrl[1]*x[4] + u_ctrl[2]*x[3] + u_ctrl[3]*x[6]),
                0.5*(-u_ctrl[1]*x[3] - u_ctrl[2]*x[4] - u_ctrl[3]*x[5]),
                2*(x[3]*x[5] + x[6]*x[4])*((u_ctrl[0]*0.1*max_thrust)/mass),
                2*(x[4]*x[5] - x[6]*x[3])*((u_ctrl[0]*0.1*max_thrust)/mass),
                (1 - 2*(x[3]**2 + x[4]**2))*((u_ctrl[0]*0.1*max_thrust)/mass) - gravity]
        return xdot

    soln = solve_ivp(lin_func, [0.0, step_horizon], np.squeeze(state_init), t_eval = [step_horizon])
    xnext = soln.y
    next_state =  np.squeeze(xnext)

    return next_state

def simulate_euler(step_horizon, t0, state_init_quat, u):

    state_init = quat2eul(state_init_quat)

    u_ctrl = np.squeeze(u)

    def lin_func(t, x):
        xdot = [x[6], x[7], x[8], u_ctrl[1] ,u_ctrl[2], u_ctrl[3],
                ((cos(x[5])*sin(x[4])*cos(x[3]) + sin(x[5])*sin(x[3]))/mass)*(u_ctrl[0]*0.1*max_thrust),
                ((sin(x[5])*sin(x[4])*cos(x[3]) - cos(x[5])*sin(x[3]))/mass)*(u_ctrl[0]*0.1*max_thrust),
                -gravity + ((cos(x[4])*cos(x[3]))/mass)*(u_ctrl[0]*0.1*max_thrust)]
        return xdot
    soln = solve_ivp(lin_func, [0.0, step_horizon], np.squeeze(state_init), t_eval = [step_horizon])
    xnext = soln.y
    next_state_euler =  np.squeeze(xnext)

    next_state_quat = eul2quat(next_state_euler)

    return next_state_quat

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

ti = 0.0
tf = 4.0

tracking = 0
mindist = 0

trajlen = int((tf-ti)/dt)

rinit_quat = Rot.from_euler('xyz', [0.0, 0.0, 0.0]).as_quat()

xinit = np.array([0.0, 0.0, 4.0,
                  rinit_quat[0], rinit_quat[1], rinit_quat[2], rinit_quat[3],
                  0.0, 0.0, 0.0,])

MPC = Quad_MPC(dt)

T_test = 5.0 + 5.0*random.random()
#omega_x_test = -1.0 + 2.0*random.random()
omega_x_test = 0.0
omega_y_test = -1.0 + 2.0*random.random()
#omega_y_test = 0.0
#omega_z_test = -2.0 + 4.0*random.random()
omega_z_test = 0.0

u0 = np.array([T_test, omega_x_test, omega_y_test, omega_z_test])

xarray_quat_test = np.zeros((trajlen+1,n))
xarray_euler_test = np.zeros((trajlen+1,n))
uarray = np.zeros((trajlen,m))

xarray_quat_test[0,:] = xinit
xarray_euler_test[0,:] = xinit

tcomp = np.zeros(trajlen)
dist = np.zeros(trajlen)

for n1 in np.arange(1,trajlen+1):
    iter = n1

    u0 = np.array([T_test, omega_x_test, omega_y_test, omega_z_test])

    xarray_quat_test[n1,:] = simulate_quat(dt, 0.0, xarray_quat_test[n1-1,:], u0)
    xarray_euler_test[n1,:] = simulate_euler(dt, 0.0, xarray_euler_test[n1-1,:], u0)
    uarray[n1-1,:] = u0

    print('Iteration = ' + str(iter))

tarray = dt*np.arange(0,iter+1)

xarray_quat = np.zeros((iter+1,n-1))
xarray_euler = np.zeros((iter+1,n-1))

xarray_quat_test = xarray_quat_test[:iter+1,:]
xarray_euler_test = xarray_euler_test[:iter+1,:]

uarray = uarray[:iter,:]

for n1 in range(iter+1):
    xarray_quat[n1,:] = quat2eul(xarray_quat_test[n1,:])
    xarray_euler[n1,:] = quat2eul(xarray_euler_test[n1,:])

plt.figure()
plt.plot(tarray, xarray_quat[:,0], label='Quaternion x')
plt.plot(tarray, xarray_euler[:,0], '--', label='Euler x')
plt.xlabel('Time (s)')
plt.ylabel('x (m)')
plt.legend()
plt.show()

plt.figure()
plt.plot(tarray, xarray_quat[:,1], label='Quaternion y')
plt.plot(tarray, xarray_euler[:,1], '--', label='Euler y')
plt.xlabel('Time (s)')
plt.ylabel('y (m)')
plt.legend()
plt.show()

plt.figure()
plt.plot(tarray, xarray_quat[:,2], label='Quaternion z')
plt.plot(tarray, xarray_euler[:,2], '--', label='Euler z')
plt.xlabel('Time (s)')
plt.ylabel('z (m)')
plt.legend()
plt.show()

plt.figure()
plt.plot(tarray, (xarray_quat[:,3]/np.pi)*180, label='Quaternion phi')
plt.plot(tarray, (xarray_quat[:,4]/np.pi)*180, label='Quaternion theta')
plt.plot(tarray, (xarray_quat[:,5]/np.pi)*180, label='Quaternion psi')
plt.plot(tarray, (xarray_euler[:,3]/np.pi)*180, '--', label='Euler phi')
plt.plot(tarray, (xarray_euler[:,4]/np.pi)*180, '--', label='Euler theta')
plt.plot(tarray, (xarray_euler[:,5]/np.pi)*180, '--', label='Euler psi')
plt.xlabel('Time (s)')
plt.ylabel('Angle (deg)')
plt.legend()
plt.show()

plt.figure()
plt.plot(tarray, xarray_quat_test[:,3], label='Quaternion qx')
plt.plot(tarray, xarray_quat_test[:,4], label='Quaternion qy')
plt.plot(tarray, xarray_quat_test[:,5], label='Quaternion qz')
plt.plot(tarray, xarray_quat_test[:,6], label='Quaternion qw')
plt.plot(tarray, xarray_euler_test[:,3], '--', label='Euler qx')
plt.plot(tarray, xarray_euler_test[:,4], '--', label='Euler qy')
plt.plot(tarray, xarray_euler_test[:,5], '--', label='Euler qz')
plt.plot(tarray, xarray_euler_test[:,6], '--', label='Euler qw')
plt.xlabel('Time (s)')
plt.ylabel('Quaternion')
plt.legend()
plt.show()

plt.figure()
plt.stairs(uarray[:,0]*0.1*max_thrust, tarray, label='Thrust')
plt.plot([tarray[0],tarray[-1]], hover_thrust*np.ones(2), '--', label='Hover Thrust')
plt.plot([tarray[0],tarray[-1]], max_thrust*np.ones(2), '--', label='Max Thrust')
plt.xlabel('Time (s)')
plt.ylabel('Thrust (N)')
plt.legend()
plt.show()

plt.figure()
plt.stairs(uarray[:,1], tarray, label='omega_x')
plt.stairs(uarray[:,2], tarray, label='omega_y')
plt.stairs(uarray[:,3], tarray, label='omega_z')
plt.xlabel('Time (s)')
plt.ylabel('Rotation Rate (rad/s)')
plt.legend()
plt.show()

plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(xarray_quat[:,0], xarray_quat[:,1], xarray_quat[:,2], label='Quaternion Trajectory')
ax.plot3D(xarray_euler[:,0], xarray_euler[:,1], xarray_euler[:,2], '--', label='Euler Trajectory')
plt.legend()
ax.set_xlabel('x, m')
ax.set_ylabel('y, m')
ax.set_zlabel('z, m')
plt.show()



