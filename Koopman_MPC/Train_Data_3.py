from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, atan, pi
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as Rot

from scipy.integrate import solve_ivp

from NMPC_CasADi_Class_1 import Quad_MPC

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
tf = 2.0

tracking = 1
mindist = 0

trajlen = int((tf-ti)/dt)

nwp = 301

nflight = 10

xarray_train = np.zeros((nflight, ((nwp-1)*trajlen)+1, n))
xarray_euler_train = np.zeros((nflight, ((nwp-1)*trajlen)+1, n-1))
uarray_train = np.zeros((nflight, ((nwp-1)*trajlen), m))

for m1 in range(nflight):
    waypoint = np.random.rand(nwp,3)
    waypoint[:,:2] = -5.0*np.ones((nwp,2)) + 10*waypoint[:,:2]
    waypoint[:,2] = 0.5*np.ones(nwp) + 4.5*waypoint[:,2]

    initpt = waypoint[0,:]
    inityaw = np.array([0.0, 0.0, 0.0])

    uinit = np.array([10*(hover_thrust/max_thrust), 0.0, 0.0, 0.0])
    for m2 in range(nwp-1):
        posinit = initpt
        postarg = waypoint[m2+1,:]

        dx = postarg[0] - posinit[0]
        dy = postarg[1] - posinit[1]

        '''
        targyaw = atan(abs(dy)/abs(dx))

        if dx >= 0:
            if dy >= 0:
                targyaw = targyaw
            else:
                targyaw = -targyaw
        else:
            if dy >= 0:
                targyaw = pi - targyaw
            else:
                targyaw = -pi + targyaw
        '''

        targyaw = -pi + 2*pi*(np.random.rand())

        rinit_quat = Rot.from_euler('xyz', inityaw).as_quat()

        xinit = np.array([posinit[0], posinit[1], posinit[2],
                        rinit_quat[0], rinit_quat[1], rinit_quat[2], rinit_quat[3],
                        0.0, 0.0, 0.0,])

        rtarg_quat = Rot.from_euler('xyz', [0.0, 0.0, targyaw]).as_quat()

        xtarg = np.array([postarg[0], postarg[1], postarg[2],
                        rtarg_quat[0], rtarg_quat[1], rtarg_quat[2], rtarg_quat[3],
                        0.0, 0.0, 0.0,])

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

        MPC = Quad_MPC(dt)

        xarray = np.zeros((trajlen+1,n))
        uarray = np.zeros((trajlen,m))

        xarray[0,:] = xinit

        tcomp = np.zeros(trajlen)
        dist = np.zeros(trajlen)

        for n1 in np.arange(1,trajlen+1):
            iter = n1

            tstart = time()
            [u0, att] = MPC.ctrl_loop(xarray[n1-1,:], xarray_targ[n1,:], u0)
            tcomp[n1-1] = time() - tstart

            control = np.zeros((m,2))
            control[:,1] = u0

            #xarray[n1,:] = MPC.simulate(dt, 0.0, xarray[n1-1,:], control)
            #xarray[n1,:] = simulate_euler(dt, 0.0, xarray[n1-1,:], control)
            #uarray[n1-1,:] = u0

            [xarray[n1,:], uarray[n1-1,:]] = simulate_euler(dt, 0.0, xarray[n1-1,:], control)

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

        xarray_euler = np.zeros((iter+1,n-1))
        xarray_euler[:,:3] = xarray[:iter+1,:3]
        xarray_euler[:,6:] = xarray[:iter+1,7:]

        xarray = xarray[:iter+1,:]

        uarray = uarray[:iter,:]

        for n1 in range(iter+1):
            xarray_euler[n1,3:6] = Rot.from_quat(xarray[n1,3:7]).as_euler('xyz')

        initpt = xarray[-1,:3]
        inityaw = xarray_euler[-1,3:6]

        uinit = uarray[-1,:]

        if m2 == 0:
            xarray_total = xarray
            xarray_euler_total = xarray_euler
            uarray_total = uarray
        else:
            xarray_total = np.vstack((xarray_total[:-1,:], xarray))
            xarray_euler_total = np.vstack((xarray_euler_total[:-1,:], xarray_euler))
            uarray_total = np.vstack((uarray_total, uarray))
    xarray_train[m1,:,:] = xarray_total
    xarray_euler_train[m1,:,:] = xarray_euler_total
    uarray_train[m1,:,:] = uarray_total

tarray = dt*np.arange(len(xarray_total[:,0]))

plt.figure()
plt.plot(tarray, xarray_euler_total[:,0])
plt.xlabel('Time (s)')
plt.ylabel('x (m)')
plt.show()

plt.figure()
plt.plot(tarray, xarray_euler_total[:,1])
plt.xlabel('Time (s)')
plt.ylabel('y (m)')
plt.show()

plt.figure()
plt.plot(tarray, xarray_euler_total[:,2])
plt.xlabel('Time (s)')
plt.ylabel('z (m)')
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
plt.stairs(uarray_total[:,0]*0.1*max_thrust, tarray, label='Thrust')
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
ax.plot3D(waypoint[:,0], waypoint[:,1], waypoint[:,2], 'o', label='Target')
plt.legend()
ax.set_xlabel('x, m')
ax.set_ylabel('y, m')
ax.set_zlabel('z, m')
plt.show()

np.save('x_quat_train_3', xarray_train)
np.save('x_eul_train_3', xarray_euler_train)
np.save('u_train_3', uarray_train)



