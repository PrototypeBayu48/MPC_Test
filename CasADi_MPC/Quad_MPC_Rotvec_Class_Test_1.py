from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, tan, pi
import matplotlib.pyplot as plt
from simulation_code import simulate

from scipy.spatial.transform import Rotation as Rot

from scipy.integrate import solve_ivp

from Quad_MPC_CasADi_Class_4 import Quad_MPC
from Quad_MPC_CasADi_Class_4 import Quad_MPC as Quad_MPC_Fast
from Quad_MPC_CasADi_Rotvec_Class_1 import Quad_MPC_Rotvec

def simulate_quat(step_horizon, t0, state_init_rotvec, u):

    state_init_rotvec = np.squeeze(state_init_rotvec)

    state_init_quat = np.zeros(10)
    state_init_quat[:3] = state_init_rotvec[:3]
    state_init_quat[7:] = state_init_rotvec[6:]
    state_init_quat[3:7] = Rot.from_rotvec(state_init_rotvec[3:6]).as_quat()

    u_ctrl = np.squeeze(u[:,1])
    s = 1/(np.linalg.norm(state_init_quat[3:7]))

    #u_ctrl[1:]  = -u_ctrl[1:]
    #u_ctrl[1:]  = u_ctrl[1:]/(np.linalg.norm(u_ctrl[1:]))

    K = 4.0

    '''
    def lin_func(t, x):
        xdot = [x[7], x[8], x[9],
                0.5*(u_ctrl[1]*x[6] - u_ctrl[2]*x[5] + u_ctrl[3]*x[4]) + K*(1 - (x[6]**2 + x[3]**2 + x[4]**2 + x[5]**2))*x[3],
                0.5*(u_ctrl[1]*x[5] + u_ctrl[2]*x[6] - u_ctrl[3]*x[3]) + K*(1 - (x[6]**2 + x[3]**2 + x[4]**2 + x[5]**2))*x[4],
                0.5*(-u_ctrl[1]*x[4] + u_ctrl[2]*x[3] + u_ctrl[3]*x[6]) + K*(1 - (x[6]**2 + x[3]**2 + x[4]**2 + x[5]**2))*x[5],
                0.5*(-u_ctrl[1]*x[3] - u_ctrl[2]*x[4] - u_ctrl[3]*x[5]) + K*(1 - (x[6]**2 + x[3]**2 + x[4]**2 + x[5]**2))*x[6],
                2*(x[3]*x[5] + x[6]*x[4])*((u_ctrl[0]*0.1*max_thrust)/mass),
                2*(x[4]*x[5] - x[6]*x[3])*((u_ctrl[0]*0.1*max_thrust)/mass),
                (x[6]**2 - x[3]**2 - x[4]**2 + x[5]**2)*((u_ctrl[0]*0.1*max_thrust)/mass) - gravity]
        return xdot
    '''

    '''
    def lin_func(t, x):
        xdot = [x[7], x[8], x[9],
                0.5*(u_ctrl[1]*x[6] - u_ctrl[2]*x[5] + u_ctrl[3]*x[4]),
                0.5*(u_ctrl[1]*x[5] + u_ctrl[2]*x[6] - u_ctrl[3]*x[3]),
                0.5*(-u_ctrl[1]*x[4] + u_ctrl[2]*x[3] + u_ctrl[3]*x[6]),
                0.5*(-u_ctrl[1]*x[3] - u_ctrl[2]*x[4] - u_ctrl[3]*x[5]),
                2*(x[3]*x[5] + x[6]*x[4])*((u_ctrl[0]*0.1*max_thrust)/mass),
                2*(x[4]*x[5] - x[6]*x[3])*((u_ctrl[0]*0.1*max_thrust)/mass),
                (x[6]**2 - x[3]**2 - x[4]**2 + x[5]**2)*((u_ctrl[0]*0.1*max_thrust)/mass) - gravity]
        return xdot
    '''

    def lin_func(t, x):
        xdot = [x[7], x[8], x[9],
                0.5*(1/(np.linalg.norm(x[3:7])))*(u_ctrl[1]*x[6] - u_ctrl[2]*x[5] + u_ctrl[3]*x[4]),
                0.5*(1/(np.linalg.norm(x[3:7])))*(u_ctrl[1]*x[5] + u_ctrl[2]*x[6] - u_ctrl[3]*x[3]),
                0.5*(1/(np.linalg.norm(x[3:7])))*(-u_ctrl[1]*x[4] + u_ctrl[2]*x[3] + u_ctrl[3]*x[6]),
                0.5*(1/(np.linalg.norm(x[3:7])))*(-u_ctrl[1]*x[3] - u_ctrl[2]*x[4] - u_ctrl[3]*x[5]),
                2*(x[3]*x[5] + x[6]*x[4])*((u_ctrl[0]*0.1*max_thrust)/mass),
                2*(x[4]*x[5] - x[6]*x[3])*((u_ctrl[0]*0.1*max_thrust)/mass),
                (x[6]**2 - x[3]**2 - x[4]**2 + x[5]**2)*((u_ctrl[0]*0.1*max_thrust)/mass) - gravity]
        return xdot

    soln = solve_ivp(lin_func, [0.0, step_horizon], np.squeeze(state_init_quat), t_eval = [step_horizon])
    xnext = soln.y
    next_state_quat =  np.squeeze(xnext)

    next_state_rotvec = np.zeros(9)
    next_state_rotvec[:3] = next_state_quat[:3]
    next_state_rotvec[6:] = next_state_quat[7:]
    next_state_rotvec[3:6] = Rot.from_quat(next_state_quat[3:7]).as_rotvec()

    return next_state_rotvec, u_ctrl

def simulate_euler(step_horizon, t0, state_init_quat, u):

    state_init = np.zeros(9)
    state_init[:3] = state_init_quat[:3]
    state_init[3:6] = Rot.from_quat(state_init_quat[3:7]).as_euler('xyz')
    state_init[6:] = state_init_quat[7:]

    u_ctrl = np.squeeze(u[:,1])

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

    next_state_quat = np.zeros(10)
    next_state_quat[:3] = next_state_euler[:3]
    next_state_quat[3:7] = Rot.from_euler('xyz', next_state_euler[3:6]).as_quat()
    next_state_quat[7:] = next_state_euler[6:]

    return next_state_quat, u_ctrl

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

#dt = 0.05
dt = 0.02

ti = 0.0
tf = 10.0

tracking = 1
mindist = 0

trajlen = int((tf-ti)/dt)

#rinit_rotvec = Rot.from_euler('xyz', [0.0, 0.0, 0.0]).as_rotvec()
#rinit_rotvec = Rot.from_euler('xyz', [0.0, 0.0, 3*pi/4]).as_rotvec()
#rinit_rotvec = Rot.from_euler('xyz', [0.0, 0.0, (160/180)*(pi)]).as_rotvec()
rinit_rotvec = Rot.from_euler('xyz', [0.0, 0.0, (175/180)*(pi)]).as_rotvec()
#rinit_rotvec = Rot.from_euler('xyz', [0.0, 0.0, (179/180)*(pi)]).as_rotvec()

xinit = np.array([0.0, 0.0, 4.0,
                  rinit_rotvec[0], rinit_rotvec[1], rinit_rotvec[2],
                  0.0, 0.0, 0.0,])

#rtarg_rotvec = Rot.from_euler('xyz', [0.0, 0.0, 0.0]).as_rotvec()
#rtarg_rotvec = Rot.from_euler('xyz', [0.0, 0.0, pi/2]).as_rotvec()
#rtarg_rotvec = Rot.from_euler('xyz', [0.0, 0.0, -3*pi/4]).as_rotvec()
#rtarg_rotvec = Rot.from_euler('xyz', [0.0, 0.0, -(160/180)*(pi)]).as_rotvec()
rtarg_rotvec = Rot.from_euler('xyz', [0.0, 0.0, -(175/180)*(pi)]).as_rotvec()
#rtarg_rotvec = Rot.from_euler('xyz', [0.0, 0.0, -(179/180)*(pi)]).as_rotvec()

xtarg = np.array([2.0, 2.0, 6.0,
                  rtarg_rotvec[0], rtarg_rotvec[1], rtarg_rotvec[2],
                  0.0, 0.0, 0.0,])

dxdt = (xtarg[0] - xinit[0])/(tf-ti)
dydt = (xtarg[1] - xinit[1])/(tf-ti)
dzdt = (xtarg[2] - xinit[2])/(tf-ti)

if tracking == 1:
    xtarg[6] = dxdt
    xtarg[7] = dydt
    xtarg[8] = dzdt

xarray_targ = np.zeros((trajlen+1,n))
xarray_targ[0,:] = xinit

for n1 in range(trajlen):
    xarray_targ[n1+1,:] = xtarg
    if tracking == 1:
        xarray_targ[n1+1,0] = xarray_targ[0,0] + dxdt*(n1+1)*dt
        xarray_targ[n1+1,1] = xarray_targ[0,1] + dydt*(n1+1)*dt
        xarray_targ[n1+1,2] = xarray_targ[0,2] + dzdt*(n1+1)*dt

u0 = np.array([10*(hover_thrust/max_thrust), 0.0, 0.0, 0.0])

MPC = Quad_MPC_Rotvec(dt)

xarray = np.zeros((trajlen+1,n))
uarray = np.zeros((trajlen,m))

xarray[0,:] = xinit

tcomp = np.zeros(trajlen)
dist = np.zeros(trajlen)

for n1 in np.arange(1,trajlen+1):
    iter = n1

    tstart = time()
    #[u0, att] = MPC.ctrl_loop(xarray[n1-1,:], xarray_targ[n1,:], u0)
    u0 = MPC.ctrl_loop(xarray[n1-1,:], xarray_targ[n1,:], u0)
    tcomp[n1-1] = time() - tstart

    control = np.zeros((m,2))
    control[:,1] = u0

    #xarray[n1,:] = MPC.simulate(dt, 0.0, xarray[n1-1,:], control)
    #xarray[n1,:] = simulate_euler(dt, 0.0, xarray[n1-1,:], control)
    #uarray[n1-1,:] = u0

    [xarray[n1,:], uarray[n1-1,:]] = simulate_quat(dt, 0.0, xarray[n1-1,:], control)
    #[xarray[n1,:], uarray[n1-1,:]] = simulate_euler(dt, 0.0, xarray[n1-1,:], control)

    #u0 = uarray[n1-1,:]

    dist[n1-1] = np.linalg.norm(xtarg[:3] - xarray[n1,:3])

    print('Iteration = ' + str(iter))
    print('Computation Time = ' + str(tcomp[n1-1]))
    print('Distance = ' + str(dist[n1-1]))

    if (dist[n1-1]) < mindist:
        print('Minimum Distance Reached')
        break
    if xarray[n1,2] < 0.0:
        print('z below 0.0')
        break

tcomp = tcomp[:iter]
dist = dist[:iter]

print('Maximum Computation Time = ' + str(np.max(tcomp)))
print('Minimum Computation Time = ' + str(np.min(tcomp)))
print('Computation Time Average = ' + str(np.mean(tcomp)))
print('Computation Time Standard Deviation = ' + str(np.std(tcomp)))

tarray = dt*np.arange(0,iter+1)

xarray_euler = np.zeros((iter+1,n-1))
xarray_euler[:,:3] = xarray[:iter+1,:3]
xarray_euler[:,6:] = xarray[:iter+1,7:]

xarray = xarray[:iter+1,:]

uarray = uarray[:iter,:]

for n1 in range(iter+1):
    xarray_euler[n1,3:6] = Rot.from_rotvec(xarray[n1,3:6]).as_euler('xyz')

fig1, ax1 = plt.subplots(3,sharex=True)
for n1 in np.arange(3):
    if n1 == 0:
        ax1[n1].plot(tarray,xarray_euler[:,n1],'b-',label='Rotvec MPC Trajectory')
        ax1[n1].plot(tarray,xarray_targ[:,n1],'c:',label='Target Trajectory')
    else:
        ax1[n1].plot(tarray,xarray_euler[:,n1],'b-')
        ax1[n1].plot(tarray,xarray_targ[:,n1],'c:')
ax1[0].set(ylabel='x (m)')
ax1[1].set(ylabel='y (m)')
ax1[2].set(ylabel='z (m)')
fig1.legend()
fig1.supxlabel('Time (s)')
plt.show()

plt.figure()
plt.plot(tarray, (xarray_euler[:,3]/np.pi)*180, label='phi')
plt.plot(tarray, (xarray_euler[:,4]/np.pi)*180, label='theta')
plt.plot(tarray, (xarray_euler[:,5]/np.pi)*180, label='psi')
plt.xlabel('Time (s)')
plt.ylabel('Angle (deg)')
plt.legend()
plt.show()

plt.figure()
plt.plot(tarray, xarray[:,3], label='p_x')
plt.plot(tarray, xarray[:,4], label='p_y')
plt.plot(tarray, xarray[:,5], label='p_z')
plt.xlabel('Time (s)')
plt.ylabel('Rotation Vector')
plt.legend()
plt.show()

plt.figure()
plt.plot(tarray, (np.linalg.norm(xarray[:,3:6], axis=1)/pi)*(180))
plt.xlabel('Time (s)')
plt.ylabel('Rotation Vector Magnitude (deg)')
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
ax.plot3D(xarray_euler[:,0], xarray_euler[:,1], xarray_euler[:,2], 'c-', label='Trajectory')
ax.plot3D(xtarg[0], xtarg[1], xtarg[2], 'o', label='Target')
plt.legend()
ax.set_xlabel('x, m')
ax.set_ylabel('y, m')
ax.set_zlabel('z, m')
plt.show()



