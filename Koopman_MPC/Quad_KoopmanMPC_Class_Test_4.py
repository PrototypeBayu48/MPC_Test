from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, tan, pi
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as Rot

from scipy.integrate import solve_ivp

from KoopmanMPC_CasADi_Class_1 import Quad_KoopmanMPC_RotVec

from Cont_func_1 import CKBF_STK

import jax

import pickle

from func_simulation import simulate_koop, simulate_euler

from func_attconv import eultorotvec, rotvectoeul

import os

def debug_func(zstep, ustep, xdata):

    datalen = len(xdata[:,0])
    predlen = len(zstep[0,:])

    xstep_norm = zstep[1:10,:]
    xstep = ((xstep_norm.T)*xscal[:,0]) + xscal[:,1]
    ustep = ((ustep.T)*uscal[:,0]) + uscal[:,1]

    xsim = np.zeros((predlen,n))
    xsim[0,:] = xstep[0,:]

    for n1 in np.arange(1,predlen):
        x0_euler = rotvectoeul(xsim[n1-1,:])
        [xnext_euler, u] = simulate_euler(dt, 0.0, x0_euler, ustep[n1-1,:], quadspec)
        xsim[n1,:] = eultorotvec(xnext_euler)
    
    tdata = dt*np.arange(datalen)
    tspan = tdata[-1] + dt*np.arange(predlen)

    xstep_euler = np.zeros((predlen,n))
    xsim_euler = np.zeros((predlen,n))
    xdata_euler = np.zeros((datalen,n))

    for n1 in range(predlen):
        xstep_euler[n1,:] = xstep[n1,:]
        xsim_euler[n1,:] = xsim[n1,:]

        xstep_euler[n1,3:6] = Rot.from_rotvec(xstep[n1,3:6]).as_euler('xyz')
        xsim_euler[n1,3:6] = Rot.from_rotvec(xsim[n1,3:6]).as_euler('xyz')

    for n1 in range(datalen):
        xdata_euler[n1,:] = xdata[n1,:]

        xdata_euler[n1,3:6] = Rot.from_rotvec(xdata[n1,3:6]).as_euler('xyz')
    
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
            ax1[n1-3].plot(tspan,xstep[:,n1],'b:',label='Koopman MPC Trajectory')
            ax1[n1-3].plot(tspan,xsim[:,n1],'c:',label='Simulation Trajectory')
            ax1[n1-3].plot(tdata,xdata[:,n1],'g-',label='Real Trajectory')
        else:
            ax1[n1-3].plot(tspan,xstep[:,n1],'b:')
            ax1[n1-3].plot(tspan,xsim[:,n1],'c:')
            ax1[n1-3].plot(tdata,xdata[:,n1],'g-')
    ax1[0].set(ylabel='e_x')
    ax1[1].set(ylabel='e_y')
    ax1[2].set(ylabel='e_z')
    fig1.legend()
    fig1.supxlabel('Time, s')
    plt.show()

    fig1, ax1 = plt.subplots(3,sharex=True)
    for n1 in np.arange(3,6):
        if n1 == 3:
            ax1[n1-3].plot(tspan,(xstep_euler[:,n1]/np.pi)*180,'b:',label='Koopman MPC Trajectory')
            ax1[n1-3].plot(tspan,(xsim_euler[:,n1]/np.pi)*180,'c:',label='Simulation Trajectory')
            ax1[n1-3].plot(tdata,(xdata_euler[:,n1]/np.pi)*180,'g-',label='Real Trajectory')
        else:
            ax1[n1-3].plot(tspan,(xstep_euler[:,n1]/np.pi)*180,'b:')
            ax1[n1-3].plot(tspan,(xsim_euler[:,n1]/np.pi)*180,'c:')
            ax1[n1-3].plot(tdata,(xdata_euler[:,n1]/np.pi)*180,'g-')
    ax1[0].set(ylabel='phi (deg)')
    ax1[1].set(ylabel='theta (deg)')
    ax1[2].set(ylabel='psi (deg)')
    fig1.legend()
    fig1.supxlabel('Time, s')
    plt.show()

def defkbf(n,m,len):
    return CKBF_STK([n, m, 13+1], 100*np.ones(len), True, jax.nn.swish)

n = 9
m = 4

quadspec_iris = [2.02, 0.011, 0.015, 0.021, 0.26, 5.84e-6, 0.06*5.84e-6, 28.3]
quadspec_crazyflie = [0.0336, 1.7e-5, 1.7e-5, 2.9e-5, 0.04, 3.1582e-10, 7.9379e-12, 0.057*9.81]

quadspec = quadspec_iris

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

#debug = 20
debug = 100

if model == 'BLO':
    koopmat_blo = np.load(os.getcwd() + '/trainmodel/matrices_dataset_6_rotvec_blo_1.npy')
    with open(os.getcwd() + '/trainmodel/net_dataset_6_rotvec_blo_1.pkl','rb') as fp:
        params_blo = pickle.load(fp)

    xscal = np.load(os.getcwd() + '/trainmodel/xscal_dataset_6_rotvec_blo_1.npy')
    uscal = np.load(os.getcwd() + '/trainmodel/uscal_dataset_6_rotvec_blo_1.npy')

    kbf = defkbf(n,m,2)
    def basis_blo(x):
        return np.array(kbf.encoder(x,params_blo))
elif model == 'Rottest':
    koopmat_blo = np.load(os.getcwd() + '/trainmodel/matrices_rotrate_rotvec_crazyflie_blo_1.npy')
    with open(os.getcwd() + '/trainmodel/net_rotrate_rotvec_crazyflie_blo_1.pkl','rb') as fp:
        params_blo = pickle.load(fp)

    xscal = np.load(os.getcwd() + '/trainmodel/xscal_rotrate_rotvec_crazyflie_blo_1.npy')
    uscal = np.load(os.getcwd() + '/trainmodel/uscal_rotrate_rotvec_crazyflie_blo_1.npy')

    kbf = defkbf(n,m,2)
    def basis_blo(x):
        return np.array(kbf.encoder(x,params_blo))

# Define Trajectory

ti = 0.0
#tf = 4.0
#tf = 8.0
tf = 10.0

tracking = 1
mindist = 0

trajlen = int((tf-ti)/dt)

rinit_rotvec = Rot.from_euler('xyz', [0.0, 0.0, 0.0]).as_rotvec()
#rinit_rotvec = Rot.from_euler('xyz', [0.0, 0.0, 3*pi/4]).as_rotvec()
#rinit_rotvec = Rot.from_euler('xyz', [0.0, 0.0, ((170/180)*pi)]).as_rotvec()

xinit = np.array([0.0, 0.0, 2.0,
                  rinit_rotvec[0], rinit_rotvec[1], rinit_rotvec[2],
                  0.0, 0.0, 0.0,])

rtarg_rotvec = Rot.from_euler('xyz', [0.0, 0.0, 0.0]).as_rotvec()
#rtarg_rotvec = Rot.from_euler('xyz', [0.0, 0.0, pi/4]).as_rotvec()
#rtarg_rotvec = Rot.from_euler('xyz', [0.0, 0.0, pi/2]).as_rotvec()
#rtarg_rotvec = Rot.from_euler('xyz', [0.0, 0.0, -3*pi/4]).as_rotvec()
#rtarg_rotvec = Rot.from_euler('xyz', [0.0, 0.0, ((-170/180)*pi)]).as_rotvec()

xtarg = np.array([0.0, 0.0, 6.0,
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

t_targ = dt*np.arange(trajlen+1)

u0 = np.array([10*(hover_thrust/max_thrust), 0.0, 0.0, 0.0])

umax = np.array([10.0, 2.0, 2.0, 2.0])
umin = np.array([0.0, -2.0, -2.0, -2.0])

MPC_blo = Quad_KoopmanMPC_RotVec(dt, koopmat_blo, basis_blo, quadspec, (umax - uscal[:,1])/uscal[:,0], (umin - uscal[:,1])/uscal[:,0],xscal,uscal)

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

    if debug >= 1:
        tstart = time()
        [upred,zpred] = MPC_blo.ctrl_loop_debug(xstart_pos, xtarg_pos, ustart)
        tcomp[n1-1] = time() - tstart
        if n1//debug == n1/debug:
            debug_func(zpred, upred, xarray[:n1,:])
        u0 = np.squeeze(upred[:,1])
        #u0 = np.squeeze(upred)
    else:
        tstart = time()
        [u0, att] = MPC_blo.ctrl_loop(xarray[n1-1,:], xarray_targ[n1,:], u0)
        tcomp[n1-1] = time() - tstart
        u0 = MPC_blo.ctrl_loop(xstart_pos, xtarg_pos, ustart)

    u0 = (u0*uscal[:,0]) + uscal[:,1]

    #control = np.zeros((m,2))
    #control[:,1] = u0
    control =  u0

    #xarray[n1,:] = MPC.simulate(dt, 0.0, xarray[n1-1,:], control)
    #xarray[n1,:] = simulate_euler(dt, 0.0, xarray[n1-1,:], control)
    #uarray[n1-1,:] = u0

    x0_euler = rotvectoeul(xarray[n1-1,:])

    [xnext_euler, uarray[n1-1,:]] = simulate_euler(dt, 0.0, x0_euler, control, quadspec)
    #[xarray[n1,:], uarray[n1-1,:]] = simulate_koop(xarray[n1-1,:], control, koopmat_blo, basis_blo, xscal, uscal)

    #u0 = uarray[n1-1,:]

    xarray[n1,:] = eultorotvec(xnext_euler)

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

xarray_euler = np.zeros((iter+1,n))
for n1 in range(iter+1):
    xarray_euler[n1,:] = rotvectoeul(xarray[n1,:])

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



