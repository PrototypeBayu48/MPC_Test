import osqp
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time

from scipy import sparse
from scipy.signal import cont2discrete
from scipy.integrate import solve_ivp
from numpy import sin,cos

from simple_pid import PID

n = 12
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

'''
mass = 0.77
Ixx = 0.0033
Iyy = 0.0033
Izz = 0.005
gravity = 9.81
l = 0.1103
ct = 8.07e-9
cq = 1.3719e-10

c1 = (Iyy - Izz)/Ixx
c2 = (Izz - Ixx)/Iyy
c3 = (Iyy - Ixx)/Izz

hover_thrust = mass*gravity
max_thrust = 12.9
min_thrust = 0.0

hover_w2 = (hover_thrust)/(4*ct)
max_w2 = (max_thrust)/(4*ct)
max_tx = ct*l*max_w2
max_ty = ct*l*max_w2
max_tz = 2*cq*max_w2
'''

Tarray = np.zeros((4,4))
Tarray[0,:] = ct
Tarray[1,1] = ct*l
Tarray[1,3] = -ct*l
Tarray[2,0] = -ct*l
Tarray[2,2] = ct*l
Tarray[3,0] = -cq
Tarray[3,1] = cq
Tarray[3,2] = -cq
Tarray[3,3] = cq

dt = 0.02

xe = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
ue = np.array([hover_thrust, 0.0, 0.0, 0.0])
#ue = np.array([0.0, 0.0, 0.0, 0.0])

def planquad_sim(x0,u,dt):
    def lin_func(t, x):
        xdot = [x[6], x[7], x[8], x[9], x[10], x[11],
                (u[0]/mass)*(cos(x[5])*sin(x[4])*cos(x[3]) + sin(x[5])*sin(x[3])), 
                (u[0]/mass)*(sin(x[5])*sin(x[4])*cos(x[3]) - cos(x[5])*sin(x[3])), 
                -gravity + (u[0]/mass)*(cos(x[4])*cos(x[3])), 
                c1*x[10]*x[11] + (u[1]/Ixx),
                c2*x[9]*x[11] + (u[2]/Iyy),
                c3*x[9]*x[10] + (u[3]/Izz)]
        return xdot
    soln = solve_ivp(lin_func, [0.0, dt], x0, t_eval = [dt])
    xnext = soln.y
    return np.squeeze(xnext)

# Constraints
umax = np.array([0.0, -max_tx, -max_ty, -max_tz])
umax = np.array([max_thrust, max_tx, max_ty, max_tz])
xmin = np.array([-20.0, -20.0, 0.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0])
xmax = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0])

# Initial and reference states
x0 = xe
xr = np.array([8.0, 8.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

nsim = 400

t_traj = dt*np.arange(nsim+1)

xarray = np.zeros((n,nsim+1))

radius = 4.0

traj_type = 'circleyaw'

if traj_type == 'straight':
    xarray[0,:] = x0[0] + ((xr[0] - x0[0])/(nsim+1))*np.arange(nsim+1)
    xarray[1,:] = x0[1] + ((xr[1] - x0[1])/(nsim+1))*np.arange(nsim+1)
    xarray[2,:] = x0[2] + ((xr[2] - x0[2])/(nsim+1))*np.arange(nsim+1)
    xarray[6,:] = ((xr[0] - x0[0])/(nsim+1))
    xarray[7,:] = ((xr[1] - x0[1])/(nsim+1))
    xarray[8,:] = ((xr[2] - x0[2])/(nsim+1))
elif traj_type == 'circle':
    xarray[0,:] = x0[0] + radius*cos(2*np.pi*(np.arange(nsim+1)/(nsim+1)))
    xarray[1,:] = x0[1] + radius*sin(2*np.pi*(np.arange(nsim+1)/(nsim+1)))
    xarray[2,:] = xe[2]
    xarray[6,:] = -radius*(2*np.pi/(nsim+1))*(sin(2*np.pi*(np.arange(nsim+1)/(nsim+1))))
    xarray[7,:] = radius*(2*np.pi/(nsim+1))*(cos(2*np.pi*(np.arange(nsim+1)/(nsim+1))))
    xarray[8,:] = 0.0
elif traj_type == 'circleyaw':
    xarray[0,:] = x0[0] + radius*cos(2*np.pi*(np.arange(nsim+1)/(nsim+1)))
    xarray[1,:] = x0[1] + radius*sin(2*np.pi*(np.arange(nsim+1)/(nsim+1)))
    xarray[2,:] = x0[2]
    xarray[6,:] = -radius*(2*np.pi/(nsim+1))*(sin(2*np.pi*(np.arange(nsim+1)/(nsim+1))))
    xarray[7,:] = radius*(2*np.pi/(nsim+1))*(cos(2*np.pi*(np.arange(nsim+1)/(nsim+1))))
    xarray[8,:] = 0.0

    xarray[5,0] = np.pi/2
    for n1 in range(nsim):
        dx = xarray[0,n1+1] - xarray[0,n1]
        dy = xarray[1,n1+1] - xarray[1,n1]
        yaw = np.arctan(abs(dy/dx))

        if dy >= 0.0:
            if dx >= 0.0:
                yaw = yaw
            else:
                yaw = np.pi - yaw
        else:
            if dx >= 0.0:
                yaw = -yaw
            else:
                yaw = -np.pi + yaw

        xarray[5,n1+1] = yaw

# Define Controller

def PIDcontrol(x0,xr,acc_des):
    ctrlinput = np.zeros(m)

    #Kppos = np.array([0.52, 0.80, 0.40])
    #Kdpos = np.array([0.00, 0.13, 0.12])
    #Kipos = np.array([0.40, 0.50, 0.30])

    Kppos = np.array([17.0, 17.0, 20.0])
    Kdpos = np.array([6.6, 6.6, 9.0])
    Kipos = np.array([0.0, 0.0, 0.0])

    ep = x0[:3] - xr[0:3]
    ev = x0[6:9] - xr[6:9]

    acc = - Kppos*ep - Kdpos*ev - Kipos*ep*dt + acc_des[:3]
    acc[2] = acc[2] + gravity

    zbd = acc/np.linalg.norm(acc)

    ctrlinput[0] = np.reshape(acc,(1,3))@np.reshape(zbd,(3,1))

    rotmat = np.array([[sin(xr[5]), -cos(xr[5])], [cos(xr[5]), sin(xr[5])]])
    omegamat = np.array([[cos(xr[5]), sin(xr[5])], [-sin(xr[5]), cos(xr[5])]])

    #xr[3:5] = (1/gravity)*rotmat@acc_des[:2]
    #xr[9:11] = (1/gravity)*omegamat@(xr[11]*acc_des[:2])

    xr[3:5] = (1/gravity)*rotmat@acc[:2]
    xr[9:11] = (1/gravity)*omegamat@(xr[11]*acc[:2])

    Kprot = np.array([190.0, 198.0, 80.0])
    Kdrot = np.array([30.0, 30.0, 17.88])
    Kirot = np.array([0.0, 0.0, 0.0])

    J = np.diag([Ixx, Iyy, Izz])

    erot = x0[3:6] - xr[3:6]
    eomega = x0[9:12] - xr[9:12]

    ctrlinput[1:] = J@(- Kprot*erot - Kdrot*eomega - Kirot*erot*dt)

    return ctrlinput

def motorlimiter(ctrlinput):
    w2 = np.abs(np.linalg.solve(Tarray,ctrlinput))

    for n1 in range(m):
        if w2[n1] > max_w2:
            w2[n1] = max_w2
    
    ctrlinput = Tarray@w2

    return ctrlinput


# Simulate in closed loop

x_result = np.zeros((n,nsim+1))
x_result[:,0] = xarray[:,0]
u_result = np.zeros((m,nsim))
u_result[:,0] = ue
t_result = t_traj
tcomp_result = np.zeros(nsim)
trackerror_result = np.zeros(nsim)

x0 = xarray[:,0]

for i in range(nsim):
    print(i)

    if i == 0:
        acc_des = np.zeros(3)
    else:
        acc_des = (xarray[6:9,i-1] - xarray[6:9,i])/dt

    t1 = time.time()

    # PID Controller
    ctrl = PIDcontrol(x0,xarray[:,i+1],acc_des)
    tcomp_result[i] = time.time() - t1

    '''
    if ctrl[0] > max_thrust:
        ctrl[0] = max_thrust
    elif ctrl[0] < min_thrust:
        ctrl[0] = min_thrust

    if ctrl[1] > max_tx:
        ctrl[1] = max_tx
    elif ctrl[1] < -max_tx:
        ctrl[1] = -max_tx

    if ctrl[2] > max_ty:
        ctrl[2] = max_ty
    elif ctrl[2] < -max_ty:
        ctrl[2] = -max_ty
    
    if ctrl[3] > max_tz:
        ctrl[3] = max_tz
    elif ctrl[3] < -max_tz:
        ctrl[3] = -max_tz
    '''

    #ctrl = motorlimiter(ctrl)

    x0 = planquad_sim(x0, ctrl, dt)

    x_result[:,i+1] = x0
    u_result[:,i] = ctrl

    trackerror_result[i] = np.linalg.norm(x0[:3]-xarray[:3,i+1])

    finaliter = i + 1

print('Average Tracking Error: ', np.mean(trackerror_result))

print(' ')

print('Average Computation Time: ', np.mean(tcomp_result))
print('Max Computation Time: ', np.max(tcomp_result))
print('Min Computation Time: ', np.min(tcomp_result))
print('Computation Time Standard Deviation: ', np.std(tcomp_result))

plt.figure()
plt.plot(trackerror_result)
plt.xlabel('Iteration')
plt.ylabel('Tracking Error (m)')
plt.show()

plt.figure()
plt.plot(t_result,x_result[0,:], label='True Trajectory')
plt.plot(t_result,xarray[0,:finaliter+1], '--', label='Target Trajectory')
plt.xlabel('Time (s)')
plt.ylabel('x (m)')
plt.legend()
plt.show()

plt.figure()
plt.plot(t_result,x_result[1,:], label='True Trajectory')
plt.plot(t_result,xarray[1,:finaliter+1], '--', label='Target Trajectory')
plt.xlabel('Time (s)')
plt.ylabel('y (m)')
plt.legend()
plt.show()

plt.figure()
plt.plot(t_result,x_result[2,:], label='True Trajectory')
plt.plot(t_result,xarray[2,:finaliter+1], '--', label='Target Trajectory')
plt.xlabel('Time (s)')
plt.ylabel('z (m)')
plt.legend()
plt.show()

plt.figure()
plt.plot(t_result,(x_result[3,:]/np.pi)*180, label='Phi Trajectory')
plt.plot(t_result,(x_result[4,:]/np.pi)*180, label='Theta Trajectory')
plt.plot(t_result,(x_result[5,:]/np.pi)*180, label='Psi Trajectory')
plt.xlabel('Time (s)')
plt.ylabel('Euler Angle (deg)')
plt.legend()
plt.show()

plt.figure()
plt.plot(t_result[:-1],u_result[0,:], label='T')
plt.plot([0,t_result[-1]],[hover_thrust,hover_thrust], '--', label='T hover')
plt.xlabel('Time (s)')
plt.ylabel('Thrust (N)')
plt.legend()
plt.show()

plt.figure()
plt.plot(t_result[:-1],u_result[1,:], label='tx')
plt.plot(t_result[:-1],u_result[2,:], label='ty')
plt.plot(t_result[:-1],u_result[3,:], label='tz')
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.show()

plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(x_result[0,:],x_result[1,:],x_result[2,:],label='Trajectory')
ax.plot3D(xarray[0,:],xarray[1,:],xarray[2,:],'--',label='Target')
plt.legend()
ax.set_xlabel('x, m')
ax.set_ylabel('y, m')
ax.set_zlabel('z, m')
plt.show()



