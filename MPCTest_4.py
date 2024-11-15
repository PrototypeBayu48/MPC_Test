import osqp
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time

from scipy import sparse
from scipy.signal import cont2discrete
from scipy.integrate import solve_ivp
from numpy import sin,cos

n = 6
m = 2

mass = 2.0
gravity = 9.81
len = 0.5
Ixx = 0.05

thrust_hover = mass*gravity

dt = 0.02

def planquad_lin(xe,ue):
    A = np.zeros((n,n))
    B = np.zeros((n,m))
    A[:3,3:] = np.eye(3)
    A[3,2] = -((ue[0] + ue[1])/mass)*(cos(xe[2]))
    A[4,2] = -((ue[0] + ue[1])/mass)*(sin(xe[2]))
    B[3,0] = -((sin(xe[2]))/mass)
    B[3,1] = -((sin(xe[2]))/mass)
    B[4,0] = ((cos(xe[2]))/mass)
    B[4,1] = ((cos(xe[2]))/mass)
    B[5,0] = -(len/Ixx)
    B[5,1] = (len/Ixx)
    return A,B

xe = np.array([0.0, 2.0, 0.0, 0.0, 0.0, 0.0])
ue = np.array([thrust_hover/2, thrust_hover/2])
#ue = np.array([0.0, 0.0])

[Alin, Blin] = planquad_lin(xe,ue)
Clin = np.zeros((2,n))
Clin[0,0] = 1
Clin[1,1] = 1
Dlin = np.zeros((2,m))

def planquad_sim(x0,u,dt):
    def lin_func(t, x):
        xdot = [x[3], x[4], x[5], -((u[0] + u[1])/mass)*(sin(x[2])), -gravity + ((u[0] + u[1])/mass)*(cos(x[2])), len*(-u[0] + u[1])/Ixx]
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
umin = np.array([0.0, 0.0])
umax = np.array([thrust_hover, thrust_hover])
xmin = np.array([-40.0, 0.0, -2*np.pi, -20.0, -20.0, -20.0])
xmax = np.array([40.0, 40.0, 2*np.pi, 20.0, 20.0, 20.0])

# Objective function
Q = sparse.diags([10.0, 10.0, 1.0, 0.0, 0.0, 0.0])
QN = Q*0
R = 0.01*sparse.eye(m)

# Initial and reference states
x0 = xe
xr = np.array([4.0, 4.0, 0.0, 0.0, 0.0, 0.0])

nsim = 400

t_traj = dt*np.arange(nsim+1)

xarray = np.zeros((n,nsim+1))

traj_type = 'circle'

if traj_type == 'straight':
    xarray[0,:] = x0[0] + ((xr[0] - x0[0])/(nsim+1))*np.arange(nsim+1)
    xarray[1,:] = x0[1] + ((xr[1] - x0[1])/(nsim+1))*np.arange(nsim+1)
elif traj_type == 'circle':
    xarray[0,:] = x0[0] + cos(2*np.pi*(np.arange(nsim+1)/(nsim+1)))
    xarray[1,:] = x0[1] + sin(2*np.pi*(np.arange(nsim+1)/(nsim+1)))

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

    x0 = planquad_sim(x0, ctrl, dt)

    x_result[:,i+1] = x0
    u_result[:,i] = ctrl

    trackerror_result[i] = np.linalg.norm(x0[:2]-xarray[:2,i+1])

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
plt.ylabel('y (m)')
plt.legend()
plt.show()

plt.figure()
plt.plot(t_result,x_result[1,:], label='True Trajectory')
plt.plot(t_result,xarray[1,:finaliter+1], '--', label='Target Trajectory')
plt.xlabel('Time (s)')
plt.ylabel('z (m)')
plt.legend()
plt.show()

plt.figure()
plt.plot(t_result,x_result[2,:])
plt.xlabel('Time (s)')
plt.ylabel('Theta (rad)')
plt.show()

plt.figure()
plt.plot(t_result[:-1],u_result[0,:], label='T1')
plt.plot(t_result[:-1],u_result[1,:], label='T2')
plt.xlabel('Time (s)')
plt.ylabel('Thrust (N)')
plt.legend()
plt.show()

plt.figure()
plt.plot(x_result[0,:],x_result[1,:], label='True Trajectory')
plt.plot(xarray[0,:],xarray[1,:], '--', label='Target Trajectory')
plt.xlabel('y (m)')
plt.ylabel('z (m)')
plt.legend()
plt.show()



