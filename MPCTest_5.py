import osqp
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time

from scipy import sparse
from scipy.signal import cont2discrete
from scipy.integrate import solve_ivp
from numpy import sin,cos

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

dt = 0.02

def planquad_lin(xe,ue):
    A = np.zeros((n,n))
    B = np.zeros((n,m))
    A[:6,6:] = np.eye(6)

    A[6,3] = (ue[0]/mass)*(-cos(xe[5])*sin(xe[4])*sin(xe[3]) + sin(xe[5])*cos(xe[3]))
    A[6,4] = (ue[0]/mass)*(cos(xe[5])*cos(xe[4])*cos(xe[3]))
    A[6,5] = (ue[0]/mass)*(-sin(xe[5])*sin(xe[4])*cos(xe[3]) + cos(xe[5])*sin(xe[3]))

    A[7,3] = (ue[0]/mass)*(-sin(xe[5])*sin(xe[4])*sin(xe[3]) - cos(xe[5])*cos(xe[3]))
    A[7,4] = (ue[0]/mass)*(sin(xe[5])*cos(xe[4])*cos(xe[3]))
    A[7,5] = (ue[0]/mass)*(cos(xe[5])*sin(xe[4])*cos(xe[3]) + sin(xe[5])*sin(xe[3]))

    A[8,3] = -(ue[0]/mass)*(cos(xe[4])*sin(xe[3]))
    A[8,4] = -(ue[0]/mass)*(sin(xe[4])*cos(xe[3]))

    A[9,10] = xe[10]*c1
    A[9,11] = xe[11]*c1

    A[10,9] = xe[9]*c2
    A[10,11] = xe[11]*c2

    A[11,9] = xe[9]*c3
    A[11,10] = xe[10]*c3

    B[6,0] = ((0.1*max_thrust)/mass)*(cos(xe[5])*sin(xe[4])*cos(xe[3]) + sin(xe[5])*sin(xe[3]))
    B[7,0] = ((0.1*max_thrust)/mass)*(sin(xe[5])*sin(xe[4])*cos(xe[3]) - cos(xe[5])*sin(xe[3]))
    B[8,0] = -((0.1*max_thrust)/mass)*(cos(xe[4])*cos(xe[3]))
    B[9,1] = (0.2*max_tx)/Ixx
    B[10,2] = (0.2*max_ty)/Iyy
    B[11,3] = (0.2*max_tz)/Izz

    return A,B

xe = np.array([0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
#ue = np.array([hover_thrust/(0.1*max_thrust), 0.0, 0.0, 0.0])
ue = np.array([0.0, 0.0, 0.0, 0.0])

[Alin, Blin] = planquad_lin(xe,ue)
Clin = np.zeros((3,n))
Clin[0,0] = 1
Clin[1,1] = 1
Clin[2,2] = 1
Dlin = np.zeros((3,m))

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
    

sysdisc = cont2discrete((Alin, Blin, Clin, Dlin), dt)

# Discrete time model
Ad = sparse.csc_matrix(sysdisc[0])
Bd = sparse.csc_matrix(sysdisc[1])
[nx, nu] = Bd.shape

# Constraints
umin = np.array([0.0, -5.0, -5.0, -5.0])
umax = np.array([10.0, 5.0, 5.0, 5.0])
xmin = np.array([-20.0, -20.0, 0.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0])
xmax = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0])

# Objective function
Q = sparse.diags([100.0, 100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
QN = Q*0
R = 0.1*sparse.eye(m)

# Initial and reference states
x0 = xe
xr = np.array([2.0, 2.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

nsim = 400

t_traj = dt*np.arange(nsim+1)

xarray = np.zeros((n,nsim+1))

traj_type = 'straight'

if traj_type == 'straight':
    xarray[0,:] = x0[0] + ((xr[0] - x0[0])/(nsim+1))*np.arange(nsim+1)
    xarray[1,:] = x0[1] + ((xr[1] - x0[1])/(nsim+1))*np.arange(nsim+1)
    xarray[2,:] = x0[2] + ((xr[2] - x0[2])/(nsim+1))*np.arange(nsim+1)
elif traj_type == 'circle':
    xarray[0,:] = x0[0] + cos(2*np.pi*(np.arange(nsim+1)/(nsim+1)))
    xarray[1,:] = x0[1] + sin(2*np.pi*(np.arange(nsim+1)/(nsim+1)))
    xarray[2,:] = xe[2]

# Prediction horizon
N = 25

def mpc_setup(x0,xr,Ad,Bd):
  # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
  # - quadratic objective
  P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                        sparse.kron(sparse.eye(N), R)], format='csc')
  # - linear objective
  q = np.hstack([np.kron(np.ones(N), -Q@xr), -QN@xr, np.zeros(N*nu)])
  # - linear dynamics
  Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
  Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
  Aeq = sparse.hstack([Ax, Bu])
  leq = np.hstack([-x0, np.zeros(N*nx)])
  ueq = leq
  # - input and state constraints
  Aineq = sparse.eye((N+1)*nx + N*nu)
  lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
  uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
  # - OSQP constraints
  A = sparse.vstack([Aeq, Aineq], format='csc')
  l = np.hstack([leq, lineq])
  u = np.hstack([ueq, uineq])

  # Create an OSQP object
  prob = osqp.OSQP()

  # Setup workspace
  #prob.setup(P, q, A, l, u, warm_starting=True)
  prob.setup(P, q, A, l, u)
  return prob

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
    t1 = time.time()
    prob = mpc_setup(x0, xarray[:,i+1], Ad, Bd)

    # Solve
    res = prob.solve()

    # Check solver status
    if res.info.status != 'solved':
        print('OSQP did not solve the problem!')
        x_result = x_result[:,:i+1]
        u_result = u_result[:,:i]
        t_result = t_result[:i+1]
        tcomp_result = tcomp_result[:i]
        trackerror_result = trackerror_result[:i]
        finaliter = i
        break

    # Apply first control input to the plant
    ctrl = res.x[-N*nu:-(N-1)*nu]
    tcomp_result[i] = time.time() - t1

    ctrl[0] = ctrl[0]*(0.1*max_thrust)
    ctrl[1] = ctrl[1]*(0.2*max_tx)
    ctrl[2] = ctrl[2]*(0.2*max_ty)
    ctrl[3] = ctrl[3]*(0.2*max_tz)

    x0 = planquad_sim(x0, ctrl, dt)

    x_result[:,i+1] = x0
    u_result[:,i] = ctrl

    trackerror_result[i] = np.linalg.norm(x0[:2]-xarray[:2,i+1])

    ctrl[0] = ctrl[0]/(0.1*max_thrust)
    ctrl[1] = ctrl[1]/(0.2*max_tx)
    ctrl[2] = ctrl[2]/(0.2*max_ty)
    ctrl[3] = ctrl[3]/(0.2*max_tz)

    [Alin, Blin] = planquad_lin(x0,ctrl)
    sysdisc = cont2discrete((Alin, Blin, Clin, Dlin), dt)
    Ad = sparse.csc_matrix(sysdisc[0])
    Bd = sparse.csc_matrix(sysdisc[1])

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



