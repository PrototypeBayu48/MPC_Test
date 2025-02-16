from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, tan, pi
import matplotlib.pyplot as plt
from simulation_code import simulate

from scipy.spatial.transform import Rotation as Rot

from scipy.integrate import solve_ivp

from Quad_MPC_CasADi_Class_2 import Quad_MPC

import random

def rotvec2eul(x_quat):
    x_euler = np.zeros(9)
    x_euler[:3] = x_quat[:3]
    x_euler[3:6] = Rot.from_rotvec(x_quat[3:6]).as_euler('xyz')
    x_euler[6:] = x_quat[6:]

    return x_euler

def eul2rotvec(x_euler):
    x_rotvec = np.zeros(9)
    x_rotvec[:3] = x_euler[:3]
    x_rotvec[3:6] = Rot.from_euler('xyz', x_euler[3:6]).as_rotvec()
    x_rotvec[6:] = x_euler[6:]
    
    return x_rotvec

def simulate_rotvec(step_horizon, t0, state_init, u):

    u_ctrl = np.squeeze(u)
    s = 1/(np.linalg.norm(state_init[3:7]))

    #u_ctrl[1:]  = -u_ctrl[1:]
    #u_ctrl[1:]  = u_ctrl[1:]/(np.linalg.norm(u_ctrl[1:]))

    '''
    def lin_func(t, x):
        xdot = [x[6], x[7], x[8],
                u_ctrl[1]*( 1 - (1/6)*( (x[4]**2) + (x[5]**2)) ) + u_ctrl[2]*( (-1/2)*(x[5]) + (1/6)*(x[3]*x[4]) ) + u_ctrl[3]*( (1/2)*(x[4]) + (1/6)*(x[3]*x[5]) ),
                u_ctrl[1]*( (1/2)*(x[5]) + (1/6)*(x[3]*x[4]) ) + u_ctrl[2]*( 1 - (1/6)*( (x[3]**2) + (x[5]**2)) ) + u_ctrl[3]*( (-1/2)*(x[3]) + (1/6)*(x[4]*x[5]) ),
                u_ctrl[1]*( (-1/2)*(x[4]) + (1/6)*(x[3]*x[5]) ) + u_ctrl[2]*( (1/2)*(x[3]) + (1/6)*(x[4]*x[5]) ) + u_ctrl[3]*( 1 - (1/6)*( (x[3]**2) + (x[4]**2)) ),
                ( (sin(np.linalg.norm(x[3:6]))/(np.linalg.norm(x[3:6])))*(x[4]) + ((1 - cos(np.linalg.norm(x[3:6])))/(np.linalg.norm(x[3:6])**2))*(x[3]*x[5]) )*((u_ctrl[0]*0.1*max_thrust)/mass),
                ( -(sin(np.linalg.norm(x[3:6]))/(np.linalg.norm(x[3:6])))*(x[3]) + ((1 - cos(np.linalg.norm(x[3:6])))/(np.linalg.norm(x[3:6])**2))*(x[4]*x[5]) )*((u_ctrl[0]*0.1*max_thrust)/mass),
                ( 1 - ((1 - cos(np.linalg.norm(x[3:6])))/(np.linalg.norm(x[3:6])**2))*( (x[3]**2) + (x[4]**2)) )*((u_ctrl[0]*0.1*max_thrust)/mass) - gravity]
        return xdot
    '''

    def lin_func(t, x):
        xdot = [x[6], x[7], x[8],
                u_ctrl[1]*( 1 - (1/6)*( (x[4]**2) + (x[5]**2)) ) + u_ctrl[2]*( (-1/2)*(x[5]) + (1/6)*(x[3]*x[4]) ) + u_ctrl[3]*( (1/2)*(x[4]) + (1/6)*(x[3]*x[5]) ),
                u_ctrl[1]*( (1/2)*(x[5]) + (1/6)*(x[3]*x[4]) ) + u_ctrl[2]*( 1 - (1/6)*( (x[3]**2) + (x[5]**2)) ) + u_ctrl[3]*( (-1/2)*(x[3]) + (1/6)*(x[4]*x[5]) ),
                u_ctrl[1]*( (-1/2)*(x[4]) + (1/6)*(x[3]*x[5]) ) + u_ctrl[2]*( (1/2)*(x[3]) + (1/6)*(x[4]*x[5]) ) + u_ctrl[3]*( 1 - (1/6)*( (x[3]**2) + (x[4]**2)) ),
                ( (x[4]) + (1/2)*(x[3]*x[5]) )*((u_ctrl[0]*0.1*max_thrust)/mass),
                ( - (x[3]) + (1/2)*(x[4]*x[5]) )*((u_ctrl[0]*0.1*max_thrust)/mass),
                ( 1 - (1/2)*( (x[3]**2) + (x[4]**2)) )*((u_ctrl[0]*0.1*max_thrust)/mass) - gravity]
        return xdot

    soln = solve_ivp(lin_func, [0.0, step_horizon], np.squeeze(state_init), t_eval = [step_horizon])
    xnext = soln.y
    next_state =  np.squeeze(xnext)

    return next_state

def simulate_euler(step_horizon, t0, state_init, u):

    u_ctrl = np.squeeze(u)

    def lin_func(t, x):
        xdot = [x[6], x[7], x[8], 
                u_ctrl[1] + u_ctrl[2]*sin(x[3])*tan(x[4]) + u_ctrl[3]*cos(x[3])*tan(x[4]), 
                u_ctrl[2]*cos(x[3]) - u_ctrl[3]*sin(x[3]), 
                u_ctrl[2]*(sin(x[3])/cos(x[4])) + u_ctrl[3]*(cos(x[3])/cos(x[4])), 
                ((cos(x[5])*sin(x[4])*cos(x[3]) + sin(x[5])*sin(x[3]))/mass)*(u_ctrl[0]*0.1*max_thrust),
                ((sin(x[5])*sin(x[4])*cos(x[3]) - cos(x[5])*sin(x[3]))/mass)*(u_ctrl[0]*0.1*max_thrust),
                -gravity + ((cos(x[4])*cos(x[3]))/mass)*(u_ctrl[0]*0.1*max_thrust)]
        return xdot
    soln = solve_ivp(lin_func, [0.0, step_horizon], np.squeeze(state_init), t_eval = [step_horizon])
    xnext = soln.y
    next_state_euler =  np.squeeze(xnext)

    return next_state_euler

n = 9
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

dt = 0.02
#dt = 0.05

ti = 0.0
tf = 2.0
#tf = 10.0

tracking = 0
mindist = 0

trajlen = int((tf-ti)/dt)

rinit_rotvec = Rot.from_euler('xyz', [0.0, 0.0, 0.0]).as_rotvec()
#rinit_rotvec = Rot.from_euler('xyz', [0.0, 0.0, pi/4]).as_rotvec()

xinit = np.array([0.0, 0.0, 4.0,
                  rinit_rotvec[0], rinit_rotvec[1], rinit_rotvec[2],
                  0.0, 0.0, 0.0,])

MPC = Quad_MPC(dt)

uarray = np.zeros((trajlen,m))

for n1 in range(trajlen):
    T_test = 5.0 + 5.0*random.random()
    omega_x_test = -4.0 + 8.0*random.random()
    #omega_x_test = 0.0
    omega_y_test = -4.0 + 8.0*random.random()
    #omega_y_test = 0.0
    omega_z_test = -4.0 + 8.0*random.random()
    #omega_z_test = 0.0

    u0 = np.array([T_test, omega_x_test, omega_y_test, omega_z_test])

    uarray[n1,:] = u0

xarray_rotvec_test = np.zeros((trajlen+1,n))
xarray_euler_test = np.zeros((trajlen+1,n))

xarray_rotvec_test[0,:] = xinit
xarray_euler_test[0,:] = rotvec2eul(xinit)

tcomp = np.zeros(trajlen)
dist = np.zeros(trajlen)

for n1 in np.arange(1,trajlen+1):
    iter = n1

    u0 = uarray[n1-1,:]

    xarray_rotvec_test[n1,:] = simulate_rotvec(dt, 0.0, xarray_rotvec_test[n1-1,:], u0)
    xarray_euler_test[n1,:] = simulate_euler(dt, 0.0, xarray_euler_test[n1-1,:], u0)

    print('Iteration = ' + str(iter))

tarray = dt*np.arange(0,iter+1)

xarray_rotvec_euler = np.zeros((iter+1,n))
xarray_euler_rotvec = np.zeros((iter+1,n))

xarray_rotvec_test = xarray_rotvec_test[:iter+1,:]
xarray_euler_test = xarray_euler_test[:iter+1,:]

uarray = uarray[:iter,:]

for n1 in range(iter+1):
    xarray_rotvec_euler[n1,:] = rotvec2eul(xarray_rotvec_test[n1,:])
    xarray_euler_rotvec[n1,:] = eul2rotvec(xarray_euler_test[n1,:])

xarray_quat_mag = np.zeros((iter+1))
xarray_euler_mag = np.zeros((iter+1))

for n1 in range(iter+1):
    xarray_quat_mag[n1] = np.linalg.norm(xarray_rotvec_test[n1,3:6])
    xarray_euler_mag[n1] = np.linalg.norm(xarray_euler_rotvec[n1,3:6])

print(xarray_quat_mag[-1])
print(xarray_euler_mag[-1])

plt.figure()
plt.plot(tarray, xarray_rotvec_test[:,0], label='Rotation Vector x')
plt.plot(tarray, xarray_euler_test[:,0], '--', label='Euler x')
plt.xlabel('Time (s)')
plt.ylabel('x (m)')
plt.legend()
plt.show()

plt.figure()
plt.plot(tarray, xarray_rotvec_test[:,1], label='Rotation Vector y')
plt.plot(tarray, xarray_euler_test[:,1], '--', label='Euler y')
plt.xlabel('Time (s)')
plt.ylabel('y (m)')
plt.legend()
plt.show()

plt.figure()
plt.plot(tarray, xarray_rotvec_test[:,2], label='Rotation Vector z')
plt.plot(tarray, xarray_euler_test[:,2], '--', label='Euler z')
plt.xlabel('Time (s)')
plt.ylabel('z (m)')
plt.legend()
plt.show()

plt.figure()
plt.plot(tarray, (xarray_rotvec_euler[:,3]/np.pi)*180, label='Rotation Vector phi')
plt.plot(tarray, (xarray_rotvec_euler[:,4]/np.pi)*180, label='Rotation Vector theta')
plt.plot(tarray, (xarray_rotvec_euler[:,5]/np.pi)*180, label='Rotation Vector psi')
plt.plot(tarray, (xarray_euler_test[:,3]/np.pi)*180, '--', label='Euler phi')
plt.plot(tarray, (xarray_euler_test[:,4]/np.pi)*180, '--', label='Euler theta')
plt.plot(tarray, (xarray_euler_test[:,5]/np.pi)*180, '--', label='Euler psi')
plt.xlabel('Time (s)')
plt.ylabel('Angle (deg)')
plt.legend()
plt.show()

plt.figure()
plt.plot(tarray, xarray_rotvec_test[:,3], label='Rotation Vector px')
plt.plot(tarray, xarray_rotvec_test[:,4], label='Rotation Vector py')
plt.plot(tarray, xarray_rotvec_test[:,5], label='Rotation Vector pz')
plt.plot(tarray, xarray_euler_rotvec[:,3], '--', label='Euler px')
plt.plot(tarray, xarray_euler_rotvec[:,4], '--', label='Euler py')
plt.plot(tarray, xarray_euler_rotvec[:,5], '--', label='Euler pz')
plt.xlabel('Time (s)')
plt.ylabel('Rotation Vector')
plt.legend()
plt.show()

plt.figure()
plt.plot(tarray, xarray_quat_mag, label='Rotation Vector Magnitude')
plt.plot(tarray, xarray_euler_mag, '--', label='Euler Magnitude')
plt.xlabel('Time (s)')
plt.ylabel('Magnitude')
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
ax.plot3D(xarray_rotvec_test[:,0], xarray_rotvec_test[:,1], xarray_rotvec_test[:,2], label='Rotation Vector Trajectory')
ax.plot3D(xarray_euler_test[:,0], xarray_euler_test[:,1], xarray_euler_test[:,2], '--', label='Euler Trajectory')
plt.legend()
ax.set_xlabel('x, m')
ax.set_ylabel('y, m')
ax.set_zlabel('z, m')
plt.show()



