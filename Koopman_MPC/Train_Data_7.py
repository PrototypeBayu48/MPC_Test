from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, atan, pi
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as Rot

from scipy.integrate import solve_ivp

from NMPC_CasADi_Class_1 import Quad_MPC_Full

from func_simulation import simulate_quat_full

from func_attconv import eultoquat, quattoeul

quadspec_crazyflie = [0.0336, 1.7e-5, 1.7e-5, 2.9e-5, 0.04, 3.1582e-10, 7.9379e-12, 0.057*9.81]
quadspec_iris = [2.02, 0.011, 0.015, 0.021, 0.26, 5.84e-6, 0.06*5.84e-6, 28.3]

quadspec = quadspec_iris

n = 13
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
#dt = 0.05

ti = 0.0
tf = 5.0

tracking = 1
mindist = 0

trajlen = int((tf-ti)/dt)

nwp = 56

nflight = 10

xarray_train = np.zeros((nflight, ((nwp-1)*trajlen)+1, 10))
xarray_euler_train = np.zeros((nflight, ((nwp-1)*trajlen)+1, 9))
uarray_train = np.zeros((nflight, ((nwp-1)*trajlen), m))

for m1 in range(nflight):
    waypoint = np.random.rand(nwp,3)
    waypoint[:,:2] = -5.0*np.ones((nwp,2)) + 10.0*waypoint[:,:2]
    waypoint[:,2] = 0.5*np.ones(nwp) + 9.5*waypoint[:,2]

    initpt = waypoint[0,:]
    inityaw = np.array([0.0, 0.0, 0.0])

    uinit = np.array([hover_w2, hover_w2, hover_w2, hover_w2])
    for m2 in range(nwp-1):
        posinit = initpt
        postarg = waypoint[m2+1,:]

        dx = postarg[0] - posinit[0]
        dy = postarg[1] - posinit[1]

        targyaw = -pi + 2*pi*(np.random.rand())
        #targyaw = -pi/8 + (pi/4)*(np.random.rand())

        rinit_quat = Rot.from_euler('xyz', inityaw).as_quat()

        xinit = np.array([posinit[0], posinit[1], posinit[2],
                        rinit_quat[0], rinit_quat[1], rinit_quat[2], rinit_quat[3],
                        0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0])

        rtarg_quat = Rot.from_euler('xyz', [0.0, 0.0, targyaw]).as_quat()

        xtarg = np.array([postarg[0], postarg[1], postarg[2],
                        rtarg_quat[0], rtarg_quat[1], rtarg_quat[2], rtarg_quat[3],
                        0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0])

        dxdt = (xtarg[0] - xinit[0])/(tf-ti)
        dydt = (xtarg[1] - xinit[1])/(tf-ti)
        dzdt = (xtarg[2] - xinit[2])/(tf-ti)

        if tracking == 1:
            xtarg[7] = dxdt
            xtarg[8] = dydt
            xtarg[9] = dzdt

        xarray_targ = np.zeros((trajlen+1,n))
        xarray_targ[0,:] = xinit

        for n1 in range(trajlen):
            xarray_targ[n1+1,:] = xtarg
            if tracking == 1:
                xarray_targ[n1+1,0] = xarray_targ[0,0] + dxdt*(n1+1)*dt
                xarray_targ[n1+1,1] = xarray_targ[0,1] + dydt*(n1+1)*dt
                xarray_targ[n1+1,2] = xarray_targ[0,2] + dzdt*(n1+1)*dt

        u0 = uinit

        MPC = Quad_MPC_Full(dt, quadspec)

        xarray = np.zeros((trajlen+1,n))
        uarray = np.zeros((trajlen,m))

        xarray[0,:] = xinit

        tcomp = np.zeros(trajlen)
        dist = np.zeros(trajlen)

        for n1 in np.arange(1,trajlen+1):
            iter = n1

            tstart = time()
            u0 = MPC.ctrl_loop(xarray[n1-1,:], xarray_targ[n1,:], u0)
            tcomp[n1-1] = time() - tstart

            #xarray[n1,:] = MPC.simulate(dt, 0.0, xarray[n1-1,:], control)
            #xarray[n1,:] = simulate_euler(dt, 0.0, xarray[n1-1,:], control)
            #uarray[n1-1,:] = u0

            [xarray[n1,:], uarray[n1-1,:]] = simulate_quat_full(dt, 0.0, xarray[n1-1,:], u0, quadspec)

            #u0 = uarray[n1-1,:]

            dist[n1-1] = np.linalg.norm(xtarg[:3] - xarray[n1,:3])

            print('Flight = ' + str(m1+1))
            print('Waypoint = ' + str(m2+1))
            print('Iteration = ' + str(iter))
            print('Computation Time = ' + str(tcomp[n1-1]))
            print('Distance = ' + str(dist[n1-1]))

            if (dist[n1-1]) < mindist:
                print('Minimum Distance Reached')
                break
            if xarray[n1,2] < 0.0:
                print('z below 0.0')
                break

        xarray = xarray[:iter+1,:]

        uarray = uarray[:iter,:]

        xarray_euler = np.zeros((iter+1,n-1))

        for n1 in range(iter+1):
            xarray_euler[n1,:] = quattoeul(xarray[n1,:], full=1)

        initpt = xarray[-1,:3]
        inityaw = xarray_euler[-1,3:6]

        uinit = uarray[-1,:]

        if m2 == 0:
            xarray_total_full = xarray
            xarray_euler_total_full = xarray_euler
            uarray_total_full = uarray
        else:
            xarray_total_full = np.vstack((xarray_total_full[:-1,:], xarray))
            xarray_euler_total_full = np.vstack((xarray_euler_total_full[:-1,:], xarray_euler))
            uarray_total_full = np.vstack((uarray_total_full, uarray))

    xarray_train[m1,:,:] = xarray_total_full[:,:10]
    xarray_euler_train[m1,:,:] = xarray_euler_total_full[:,:9]

    thrust_total = ct*np.sum(uarray_total_full, axis=1)
    uarray_train[m1,:,0] = thrust_total
    uarray_train[m1,:,1:] = xarray_total_full[:-1,10:]

tarray = dt*np.arange(len(xarray_total_full[:,0]))

fig1, ax1 = plt.subplots(3,sharex=True)
for n1 in np.arange(3):
    if n1 == 0:
        ax1[n1].plot(tarray,xarray_total_full[:,n1],'b-',label='MPC Trajectory')
    else:
        ax1[n1].plot(tarray,xarray_total_full[:,n1],'b-')
ax1[0].set(ylabel='x (m)')
ax1[1].set(ylabel='y (m)')
ax1[2].set(ylabel='z (m)')
fig1.legend()
fig1.supxlabel('Time (s)')
plt.show()

plt.figure()
plt.plot(tarray, (xarray_euler_total_full[:,3]/np.pi)*180, label='phi')
plt.plot(tarray, (xarray_euler_total_full[:,4]/np.pi)*180, label='theta')
plt.plot(tarray, (xarray_euler_total_full[:,5]/np.pi)*180, label='psi')
plt.xlabel('Time (s)')
plt.ylabel('Angle (deg)')
plt.legend()
plt.show()

plt.figure()
plt.plot(tarray, xarray_total_full[:,3], label='qx')
plt.plot(tarray, xarray_total_full[:,4], label='qy')
plt.plot(tarray, xarray_total_full[:,5], label='qz')
plt.plot(tarray, xarray_total_full[:,6], label='qw')
plt.xlabel('Time (s)')
plt.ylabel('Quaternion')
plt.legend()
plt.show()

plt.figure()
plt.stairs(uarray_total_full[:,0], tarray, label='w2_1')
plt.stairs(uarray_total_full[:,1], tarray, label='w2_2')
plt.stairs(uarray_total_full[:,2], tarray, label='w2_3')
plt.stairs(uarray_total_full[:,3], tarray, label='w2_4')
plt.plot([tarray[0],tarray[-1]], max_w2*np.ones(2), '--', label='Max w2')
plt.xlabel('Time (s)')
plt.ylabel('w^2 ((rad/s)^2)')
plt.legend()
plt.show()

plt.figure()
plt.stairs(thrust_total, tarray, label='Thrust')
plt.plot([tarray[0],tarray[-1]], hover_thrust*np.ones(2), '--', label='Hover Thrust')
plt.plot([tarray[0],tarray[-1]], max_thrust*np.ones(2), '--', label='Max Thrust')
plt.xlabel('Time (s)')
plt.ylabel('Thrust (N)')
plt.legend()
plt.show()

plt.figure()
plt.stairs(xarray_total_full[:-1,10], tarray, label='omega_x')
plt.stairs(xarray_total_full[:-1,11], tarray, label='omega_y')
plt.stairs(xarray_total_full[:-1,12], tarray, label='omega_z')
plt.xlabel('Time (s)')
plt.ylabel('Rotation Rate (rad/s)')
plt.legend()
plt.show()

plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(xarray_euler_total_full[:,0], xarray_euler_total_full[:,1], xarray_euler_total_full[:,2], 'c-', label='Trajectory')
ax.plot3D(waypoint[:,0], waypoint[:,1], waypoint[:,2], 'o', label='Target')
plt.legend()
ax.set_xlabel('x, m')
ax.set_ylabel('y, m')
ax.set_zlabel('z, m')
plt.show()

np.save('x_quat_train_7', xarray_train)
np.save('x_eul_train_7', xarray_euler_train)
np.save('u_train_quad_7', uarray_train)



