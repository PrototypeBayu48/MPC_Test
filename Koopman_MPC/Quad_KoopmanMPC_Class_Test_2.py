from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, pi
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as Rot

from scipy.integrate import solve_ivp

from KoopmanMPC_CasADi_Class_1 import Quad_KoopmanMPC_Euler

from Cont_func_1 import CKBF_STK

import jax

import pickle

def simulate_euler(step_horizon, t0, state_init, u):

    #u_ctrl = np.squeeze(u[:,1])
    u_ctrl = u

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

    return next_state_euler, u_ctrl

def simulate_koop(state_init, u, koopmat, basis, xscal, uscal):

    u_ctrl = np.squeeze(u[:,1])

    u_ctrl_norm = (u_ctrl-uscal[:,1])/uscal[:,0]

    state_init_norm = (state_init-xscal[:,1])/xscal[:,0]
    state_init_z = basis(state_init_norm)

    next_state_z = state_init_z + dt*(koopmat[0,:,:] + koopmat[1,:,:]*u_ctrl_norm[0] + koopmat[2,:,:]*u_ctrl_norm[1] + koopmat[3,:,:]*u_ctrl_norm[2] + koopmat[4,:,:]*u_ctrl_norm[3])@state_init_z

    next_state_euler = (next_state_z[1:10]*xscal[:,0]) + xscal[:,1]

    return next_state_euler, u_ctrl

def debug_func(zstep, ustep, xdata):

    datalen = len(xdata[:,0])
    predlen = len(zstep[0,:])

    xstep_norm = zstep[1:10,:]
    xstep = ((xstep_norm.T)*xscal[:,0]) + xscal[:,1]
    ustep = ((ustep.T)*uscal[:,0]) + uscal[:,1]

    xsim = np.zeros((predlen,n))
    xsim[0,:] = xstep[0,:]

    for n1 in np.arange(1,predlen):
        [xsim[n1,:],u] = simulate_euler(dt, 0.0, xsim[n1-1,:], ustep[n1-1,:])
    
    tdata = dt*np.arange(datalen)
    tspan = tdata[-1] + dt*np.arange(predlen)
    
    fig1, ax1 = plt.subplots(3,sharex=True)
    for n1 in np.arange(3):
        if n1 == 0:
            ax1[n1].plot(tspan,xstep[:,n1],'b:',label='Koopman MPC Trajectory')
            ax1[n1].plot(tspan,xsim[:,n1],'c:',label='Simulation Trajectory')
            ax1[n1].plot(tdata,xdata[:,n1],'g-',label='Real Trajectory')
        else:
            ax1[n1].plot(tspan,xstep[:,n1],'b:')
            ax1[n1].plot(tspan,xsim[:,n1],'c:')
            ax1[n1].plot(tdata,xdata[:,n1],'g-')
    ax1[0].set(ylabel='x (m)')
    ax1[1].set(ylabel='y (m)')
    ax1[2].set(ylabel='z (m)')
    fig1.legend()
    fig1.supxlabel('Time, s')
    plt.show()

    fig1, ax1 = plt.subplots(3,sharex=True)
    for n1 in np.arange(3,6):
        if n1 == 3:
            ax1[n1-3].plot(tspan,(xstep[:,n1]/np.pi)*180,'b:',label='Koopman MPC Trajectory')
            ax1[n1-3].plot(tspan,(xsim[:,n1]/np.pi)*180,'c:',label='Simulation Trajectory')
            ax1[n1-3].plot(tdata,(xdata[:,n1]/np.pi)*180,'g-',label='Real Trajectory')
        else:
            ax1[n1-3].plot(tspan,(xstep[:,n1]/np.pi)*180,'b:')
            ax1[n1-3].plot(tspan,(xsim[:,n1]/np.pi)*180,'c:')
            ax1[n1-3].plot(tdata,(xdata[:,n1]/np.pi)*180,'g-')
    ax1[0].set(ylabel='phi (deg)')
    ax1[1].set(ylabel='theta (deg)')
    ax1[2].set(ylabel='psi (deg)')
    fig1.legend()
    fig1.supxlabel('Time, s')
    plt.show()

def defkbf(n,m):
    return CKBF_STK([n, m, 13+1], 100*np.ones(2), True, jax.nn.swish)

n = 9
m = 4

quadspec = [0.0336, 1.7e-5, 1.7e-5, 2.9e-5, 0.04, 3.1582e-10, 7.9379e-12, 0.057*9.81]

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
#dt = 0.05

model = 'BLO'

model_norm = 1

debug = 1

if model == 'BLO':
    if model_norm == 1:
        koopmat_blo = np.load('matrices_rotrate_crazyflie_blo_4.npy')
        with open('net_rotrate_crazyflie_blo_4.pkl','rb') as fp:
            params_blo = pickle.load(fp)

        xscal = np.load('xscal_rotrate_crazyflie_blo_4.npy')
        uscal = np.load('uscal_rotrate_crazyflie_blo_4.npy')

        kbf = defkbf(n,m)
        def basis_blo(x):
            return np.array(kbf.encoder(x,params_blo))
    else:
        koopmat_blo = np.load('matrices_rotrate_crazyflie_blo_3.npy')
        with open('net_rotrate_crazyflie_blo_3.pkl','rb') as fp:
            params_blo = pickle.load(fp)

        xscal = np.zeros((n,2))
        xscal[:,0] = 1

        uscal = np.zeros((n,2))
        uscal[:,0] = 1

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

# Define Trajectory

ti = 0.0
tf = 4.0
#tf = 8.0

tracking = 1
mindist = 0

trajlen = int((tf-ti)/dt)

#yawinit = 0.0
yawinit = (3*np.pi)/4

#yawtarg = 0.0
#yawtarg = np.pi/2
yawtarg = (-3*np.pi)/4

xinit = np.array([0.0, 0.0, 0.2,
                  0.0, 0.0, yawinit,
                  0.0, 0.0, 0.0,])

xtarg = np.array([0.5, 0.5, 1.0,
                  0.0, 0.0, yawtarg,
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

t_targ = dt*np.arange(trajlen+1)

u0 = np.array([10*(hover_thrust/max_thrust), 0.0, 0.0, 0.0])

umax = np.array([10.0, 2.0, 2.0, 2.0])
umin = np.array([0.0, -2.0, -2.0, -2.0])

MPC_blo = Quad_KoopmanMPC_Euler(dt, koopmat_blo, basis_blo, quadspec, (umax - uscal[:,1])/uscal[:,0], (umin - uscal[:,1])/uscal[:,0],xscal,uscal)

xarray = np.zeros((trajlen+1,n))
uarray = np.zeros((trajlen,m))

xarray[0,:] = xinit

tcomp = np.zeros(trajlen)
dist = np.zeros(trajlen)

for n1 in np.arange(1,trajlen+1):
    iter = n1

    xstart_pos = (xarray[n1-1,:] - xscal[:,1])/xscal[:,0]
    xtarg_pos = (xarray_targ[n1,:] - xscal[:,1])/xscal[:,0]

    ustart = (u0 - uscal[:,1])/uscal[:,0]

    if debug == 1:
        [upred,zpred] = MPC_blo.ctrl_loop_debug(xstart_pos, xtarg_pos, ustart)
        if n1//10 == n1/10:
            debug_func(zpred, upred, xarray[:n1,:])
        u0 = np.squeeze(upred[:,1])
    else:
        tstart = time()
        #[u0, att] = MPC_blo.ctrl_loop(xarray[n1-1,:], xarray_targ[n1,:], u0)
        u0 = MPC_blo.ctrl_loop(xstart_pos, xtarg_pos, ustart)
        tcomp[n1-1] = time() - tstart

    u0 = (u0*uscal[:,0]) + uscal[:,1]

    #control = np.zeros((m,2))
    #control[:,1] = u0
    control =  u0

    #xarray[n1,:] = MPC.simulate(dt, 0.0, xarray[n1-1,:], control)
    #xarray[n1,:] = simulate_euler(dt, 0.0, xarray[n1-1,:], control)
    #uarray[n1-1,:] = u0

    [xarray[n1,:], uarray[n1-1,:]] = simulate_euler(dt, 0.0, xarray[n1-1,:], control)
    #[xarray[n1,:], uarray[n1-1,:]] = simulate_koop(xarray[n1-1,:], control, koopmat_blo, basis_blo, xscal, uscal)

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

print('Computation Time Average = ' + str(np.mean(tcomp)))
print('Computation Time Standard Deviation = ' + str(np.std(tcomp)))

tarray = dt*np.arange(0,iter+1)

xarray_euler = xarray[:iter+1,:]
uarray = uarray[:iter,:]

fig1, ax1 = plt.subplots(3,sharex=True)
for n1 in np.arange(3):
    if n1 == 0:
        ax1[n1].plot(tarray,xarray_euler[:,n1],'b-',label='Koopman MPC Trajectory')
        ax1[n1].plot(t_targ,xarray_targ[:,n1],'c:',label='Target Trajectory')
        ax1[n1].plot([tarray[0],tarray[-1]],[xtarg[n1],xtarg[n1]],'k:',label='Target Position')
    else:
        ax1[n1].plot(tarray,xarray_euler[:,n1],'b-')
        ax1[n1].plot(t_targ,xarray_targ[:,n1],'c:')
        ax1[n1].plot([tarray[0],tarray[-1]],[xtarg[n1],xtarg[n1]],'k:')
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

'''
plt.figure()
plt.plot(tarray, xarray[:,3], label='qx')
plt.plot(tarray, xarray[:,4], label='qy')
plt.plot(tarray, xarray[:,5], label='qz')
plt.plot(tarray, xarray[:,6], label='qw')
plt.xlabel('Time (s)')
plt.ylabel('Quaternion')
plt.legend()
plt.show()
'''

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



