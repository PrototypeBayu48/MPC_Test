from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, pi
import matplotlib.pyplot as plt
from simulation_code import simulate

from scipy.spatial.transform import Rotation as Rot

from scipy.integrate import solve_ivp

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

dt = 0.02

step_horizon = dt   # time between steps in seconds
N = 25              # number of look ahead steps
#N = 15              # number of look ahead steps
#sim_time = 2.0      # simulation time
sim_time = 5.0      # simulation time

# specs
x_init = 0.0
y_init = 0.0
z_init = 5.0

rinit_quat = Rot.from_euler('xyz', [0.0, 0.0, 0.0]).as_quat()

q_x_init = rinit_quat[0]
q_y_init = rinit_quat[1]
q_z_init = rinit_quat[2]
q_w_init = rinit_quat[3]

dx_init = 0.0
dy_init = 0.0
dz_init = 0.0


x_targ = 2.0
y_targ = 2.0
z_targ = 10.0

rtarg_quat = Rot.from_euler('xyz', [0.0, 0.0, 0.0]).as_quat()
#rtarg_quat = Rot.from_euler('xyz', [0.0, 0.0, np.pi/2]).as_quat()

q_x_targ = rtarg_quat[0]
q_y_targ = rtarg_quat[1]
q_z_targ = rtarg_quat[2]
q_w_targ = rtarg_quat[3]

dx_targ = 0.0
dy_targ = 0.0
dz_targ = 0.0

'''
def shift_timestep(step_horizon, t0, state_init, u, f):
    f_value = f(state_init, u[:, 0])
    next_state = ca.DM.full(state_init + (step_horizon * f_value))

    t0 = t0 + step_horizon
    u0 = ca.horzcat(
        u[:, 1:],
        ca.reshape(u[:, -1], -1, 1)
    )

    return t0, next_state, u0
'''

def shift_timestep(step_horizon, t0, state_init, u, f):

    u_ctrl = np.squeeze(u[:,1])

    def lin_func(t, x):
        xdot = [x[7], x[8], x[9],
                0.5*(u_ctrl[1]*x[6] + u_ctrl[2]*x[5] - u_ctrl[3]*x[4]),
                0.5*(-u_ctrl[1]*x[5] + u_ctrl[2]*x[6] + u_ctrl[3]*x[3]),
                0.5*(u_ctrl[1]*x[4] - u_ctrl[2]*x[3] + u_ctrl[3]*x[6]),
                0.5*(-u_ctrl[1]*x[3] - u_ctrl[2]*x[4] - u_ctrl[3]*x[5]),
                2*(x[3]*x[5] - x[6]*x[4])*((u_ctrl[0]*0.1*max_thrust)/mass),
                2*(x[4]*x[5] + x[6]*x[3])*((u_ctrl[0]*0.1*max_thrust)/mass),
                (1 - 2*(x[3]**2 + x[4]**2))*((u_ctrl[0]*0.1*max_thrust)/mass) - gravity]
        return xdot
    soln = solve_ivp(lin_func, [0.0, step_horizon], np.squeeze(state_init), t_eval = [step_horizon])
    xnext = soln.y
    next_state =  np.squeeze(xnext)

    t0 = t0 + step_horizon
    u0 = ca.horzcat(
        u[:, 1:],
        ca.reshape(u[:, -1], -1, 1)
    )

    return t0, next_state, u0

def DM2Arr(dm):
    return np.array(dm.full())


# state symbolic variables
x = ca.SX.sym('x')
y = ca.SX.sym('y')
z = ca.SX.sym('z')
q_x = ca.SX.sym('q_x')
q_y = ca.SX.sym('q_y')
q_z = ca.SX.sym('q_z')
q_w = ca.SX.sym('q_w')
dx = ca.SX.sym('dx')
dy = ca.SX.sym('dy')
dz = ca.SX.sym('dz')
states = ca.vertcat(
    x,
    y,
    z,
    q_x,
    q_y,
    q_z,
    q_w,
    dx,
    dy,
    dz
)
n_states = states.numel()

# control symbolic variables
Th = ca.SX.sym('Th')
omega_x = ca.SX.sym('omega_x')
omega_y = ca.SX.sym('omega_y')
omega_z = ca.SX.sym('omega_z')
controls = ca.vertcat(
    Th,
    omega_x,
    omega_y,
    omega_z
)
n_controls = controls.numel()

# matrix containing all states over all time steps +1 (each column is a state vector)
X = ca.SX.sym('X', n_states, N + 1)

# matrix containing all control actions over all time steps (each column is an action vector)
U = ca.SX.sym('U', n_controls, N)

# coloumn vector for storing initial state and target state
P = ca.SX.sym('P', n_states + n_states)

# state weights matrix (Q_X, Q_Y, Q_THETA)
Q = ca.diagcat(10.0, 10.0, 10.0, 
               10.0, 10.0, 10.0, 
               0.2, 0.2, 0.2)

# controls weights matrix
#R = ca.diagcat(1e-2, 1e-2, 1e-2, 1e-2)
R = ca.diagcat(5e-3, 5e-3, 5e-3, 5e-3)

# discretization model (e.g. x2 = f(x1, v, t) = x1 + v * dt)
quad_RHS = ca.vertcat(
    dx,
    dy,
    dz,
    0.5*(omega_x*q_w + omega_y*q_z - omega_z*q_y),
    0.5*(-omega_x*q_z + omega_y*q_w + omega_z*q_x),
    0.5*(omega_x*q_y - omega_y*q_x + omega_z*q_w),
    0.5*(-omega_x*q_x - omega_y*q_y - omega_z*q_z),
    2*(q_x*q_z - q_w*q_y)*((Th*0.1*max_thrust)/mass),
    2*(q_y*q_z + q_w*q_x)*((Th*0.1*max_thrust)/mass),
    (1 - 2*(q_x**2 + q_y**2))*((Th*0.1*max_thrust)/mass) - gravity
)

# RHS = states + J @ controls * step_horizon  # Euler discretization
RHS = quad_RHS
# maps controls from [va, vb, vc, vd].T to [vx, vy, omega].T
f = ca.Function('f', [states, controls], [RHS])


cost_fn = 0  # cost function
g = X[:, 0] - P[:n_states]  # constraints in the equation

'''
# runge kutta
for k in range(N):
    st = X[:, k]
    con = U[:, k]
    cost_fn = cost_fn \
        + (st[:3] - P[n_states:n_states+3]).T @ Q[:3,:3] @ (st[:3] - P[n_states:n_states+3]) \
        + Q[3,3]*(st[3]*P[16] - st[4]*P[15] + st[5]*P[14] - st[6]*P[13]).T @ (st[3]*P[16] - st[4]*P[15] + st[5]*P[14] - st[6]*P[13]) \
        + Q[4,4]*(st[3]*P[15] + st[4]*P[16] - st[5]*P[13] - st[6]*P[13]).T @ (st[3]*P[15] + st[4]*P[16] - st[5]*P[13] - st[6]*P[13]) \
        + Q[5,5]*(-st[3]*P[14] + st[4]*P[13] + st[5]*P[16] - st[6]*P[15]).T @ (-st[3]*P[14] + st[4]*P[13] + st[5]*P[16] - st[6]*P[15]) \
        + (st[7:] - P[n_states+7:]).T @ Q[6:,6:] @ (st[7:] - P[n_states+7:]) \
        + con.T @ R @ con
    st_next = X[:, k+1]
    k1 = f(st, con)
    k2 = f(st + step_horizon/2*k1, con)
    k3 = f(st + step_horizon/2*k2, con)
    k4 = f(st + step_horizon * k3, con)
    st_next_RK4 = st + (step_horizon / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    g = ca.vertcat(g, st_next - st_next_RK4)
'''


# euler
for k in range(N):
    st = X[:, k]
    con = U[:, k]
    cost_fn = cost_fn \
        + (st[:3] - P[n_states:n_states+3]).T @ Q[:3,:3] @ (st[:3] - P[n_states:n_states+3]) \
        + Q[3,3]*(st[3]*P[16] - st[4]*P[15] + st[5]*P[14] - st[6]*P[13]).T @ (st[3]*P[16] - st[4]*P[15] + st[5]*P[14] - st[6]*P[13]) \
        + Q[4,4]*(st[3]*P[15] + st[4]*P[16] - st[5]*P[13] - st[6]*P[13]).T @ (st[3]*P[15] + st[4]*P[16] - st[5]*P[13] - st[6]*P[13]) \
        + Q[5,5]*(-st[3]*P[14] + st[4]*P[13] + st[5]*P[16] - st[6]*P[15]).T @ (-st[3]*P[14] + st[4]*P[13] + st[5]*P[16] - st[6]*P[15]) \
        + (st[7:] - P[n_states+7:]).T @ Q[6:,6:] @ (st[7:] - P[n_states+7:]) \
        + con.T @ R @ con
    st_next = X[:, k+1]
    k1 = f(st, con)
    st_next_RK4 = st + (step_horizon)*(k1)
    g = ca.vertcat(g, st_next - st_next_RK4)


OPT_variables = ca.vertcat(
    X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
    U.reshape((-1, 1))
)
nlp_prob = {
    'f': cost_fn,
    'x': OPT_variables,
    'g': g,
    'p': P
}

opts = {
    'ipopt': {
        'max_iter': 2000,
        'print_level': 0,
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6,
        'linear_solver': 'MA27'
    },
    'print_time': 0
}

solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

lbx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))
ubx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))

lbx[0: n_states*(N+1): n_states] = -ca.inf      # x lower bound
lbx[1: n_states*(N+1): n_states] = -ca.inf      # y lower bound
lbx[2: n_states*(N+1): n_states] = -ca.inf      # z lower bound
lbx[3: n_states*(N+1): n_states] = -1.0         # q_x lower bound
lbx[4: n_states*(N+1): n_states] = -1.0         # q_y lower bound
lbx[5: n_states*(N+1): n_states] = -1.0         # q_z lower bound
lbx[6: n_states*(N+1): n_states] = -1.0         # q_w lower bound
lbx[7: n_states*(N+1): n_states] = -ca.inf      # dx lower bound
lbx[8: n_states*(N+1): n_states] = -ca.inf      # dy lower bound
lbx[9: n_states*(N+1): n_states] = -ca.inf      # dz lower bound

ubx[0: n_states*(N+1): n_states] = ca.inf      # x upper bound
ubx[1: n_states*(N+1): n_states] = ca.inf      # y upper bound
ubx[2: n_states*(N+1): n_states] = ca.inf      # z upper bound
ubx[3: n_states*(N+1): n_states] = 1.0         # q_x upper bound
ubx[4: n_states*(N+1): n_states] = 1.0         # q_y upper bound
ubx[5: n_states*(N+1): n_states] = 1.0         # q_z upper bound
ubx[6: n_states*(N+1): n_states] = 1.0         # q_w upper bound
ubx[7: n_states*(N+1): n_states] = ca.inf      # dx upper bound
ubx[8: n_states*(N+1): n_states] = ca.inf      # dy upper bound
ubx[9: n_states*(N+1): n_states] = ca.inf      # dz upper bound

lbx[n_states*(N+1): n_states*(N+1) + n_controls*N: n_controls] = 0.0                # Th lower bound
lbx[n_states*(N+1) + 1: n_states*(N+1) + n_controls*N: n_controls] = -2.0           # Th lower bound
lbx[n_states*(N+1) + 2: n_states*(N+1) + n_controls*N: n_controls] = -2.0           # Th lower bound
lbx[n_states*(N+1) + 3: n_states*(N+1) + n_controls*N: n_controls] = -2.0           # Th lower bound

ubx[n_states*(N+1): n_states*(N+1) + n_controls*N: n_controls] = 10.0               # Th upper bound
ubx[n_states*(N+1) + 1: n_states*(N+1) + n_controls*N: n_controls] = 2.0            # Th upper bound
ubx[n_states*(N+1) + 2: n_states*(N+1) + n_controls*N: n_controls] = 2.0            # Th upper bound
ubx[n_states*(N+1) + 3: n_states*(N+1) + n_controls*N: n_controls] = 2.0            # Th upper bound


args = {
    'lbg': ca.DM.zeros((n_states*(N+1), 1)),  # constraints lower bound
    'ubg': ca.DM.zeros((n_states*(N+1), 1)),  # constraints upper bound
    'lbx': lbx,
    'ubx': ubx
}

t0 = 0
state_init = ca.DM([x_init, y_init, z_init, q_x_init, q_y_init, q_z_init, q_w_init, dx_init, dy_init, dz_init])         # initial state
state_target = ca.DM([x_targ, y_targ, z_targ, q_x_targ, q_y_targ, q_z_targ, q_w_targ, dx_targ, dy_targ, dz_targ])       # target state

# xx = DM(state_init)
t = ca.DM(t0)

ctrl_init = ca.DM([10*(hover_thrust/max_thrust), 0.0, 0.0, 0.0])

#u0 = ca.DM.zeros((n_controls, N))  # initial control
u0 = ca.repmat(ctrl_init, 1, N)         # initial control
X0 = ca.repmat(state_init, 1, N+1)         # initial state full


mpc_iter = 0
cat_states = DM2Arr(X0)
cat_controls = DM2Arr(u0[:, 0])
times = np.array([[0]])


###############################################################################

if __name__ == '__main__':
    main_loop = time()  # return time in sec
    while (state_init[2] > 0.0) and (mpc_iter * step_horizon < sim_time) and (ca.norm_2(state_init[:3] - state_target[:3]) > 0.02):
        t1 = time()
        args['p'] = ca.vertcat(
            state_init,    # current state
            state_target   # target state
        )
        # optimization variable current state
        args['x0'] = ca.vertcat(
            ca.reshape(X0, n_states*(N+1), 1),
            ca.reshape(u0, n_controls*N, 1)
        )

        sol = solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )

        u = ca.reshape(sol['x'][n_states * (N + 1):], n_controls, N)
        X0 = ca.reshape(sol['x'][: n_states * (N+1)], n_states, N+1)

        cat_states = np.dstack((
            cat_states,
            DM2Arr(X0)
        ))

        cat_controls = np.vstack((
            cat_controls,
            DM2Arr(u[:, 0])
        ))
        t = np.vstack((
            t,
            t0
        ))

        t0, state_init, u0 = shift_timestep(step_horizon, t0, state_init, u, f)

        # print(X0)
        X0 = ca.horzcat(
            X0[:, 1:],
            ca.reshape(X0[:, -1], -1, 1)
        )

        # xx ...
        t2 = time()
        print(mpc_iter)
        print(t2-t1)
        print(ca.norm_2(state_init[:3] - state_target[:3]))
        times = np.vstack((
            times,
            t2-t1
        ))

        mpc_iter = mpc_iter + 1

    main_loop_time = time()
    ss_error = ca.norm_2(state_init[:3] - state_target[:3])

    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('avg iteration time: ', np.array(times).mean() * 1000, 'ms')
    print('final error: ', ss_error)

result_len = len(cat_states[0,0,:])
t_result = dt*np.arange(result_len)

euler_result = np.zeros((result_len,3))

for n1 in range(result_len):
    euler_result[n1,:] = Rot.from_quat(cat_states[3:7,0,n1]).as_euler('xyz')

plt.figure()
plt.plot(t_result,cat_states[0,0,:])
plt.xlabel('Time (s)')
plt.ylabel('x (m)')
plt.show()

plt.figure()
plt.plot(t_result,cat_states[1,0,:])
plt.xlabel('Time (s)')
plt.ylabel('y (m)')
plt.show()

plt.figure()
plt.plot(t_result,cat_states[2,0,:])
plt.xlabel('Time (s)')
plt.ylabel('z (m)')
plt.show()

plt.figure()
plt.plot(t_result,(euler_result[:,0]/np.pi)*180, label='phi')
plt.plot(t_result,(euler_result[:,1]/np.pi)*180, label='theta')
plt.plot(t_result,(euler_result[:,2]/np.pi)*180, label='psi')
plt.xlabel('Time (s)')
plt.ylabel('Angle (deg)')
plt.legend()
plt.show()

plt.figure()
plt.plot(t_result,cat_controls[0::4]*0.1*max_thrust)
plt.xlabel('Time (s)')
plt.ylabel('Thrust (N)')
plt.show()

plt.figure()
plt.plot(t_result,cat_controls[1::4], label='omega_x')
plt.plot(t_result,cat_controls[2::4], label='omega_y')
plt.plot(t_result,cat_controls[3::4], label='omega_z')
plt.xlabel('Time (s)')
plt.ylabel('Rotation Rate (rad/s)')
plt.legend()
plt.show()

plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(cat_states[0,0,:], cat_states[1,0,:], cat_states[2,0,:], 'c-', label='Trajectory')
ax.plot3D(x_targ, y_targ, z_targ, 'o', label='Target')
plt.legend()
ax.set_xlabel('x, m')
ax.set_ylabel('y, m')
ax.set_zlabel('z, m')
plt.show()



