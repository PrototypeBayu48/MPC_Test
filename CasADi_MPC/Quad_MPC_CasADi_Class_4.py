from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, pi
import matplotlib.pyplot as plt
from simulation_code import simulate

from scipy.spatial.transform import Rotation as Rot

from scipy.integrate import solve_ivp


class Quad_MPC:
    def __init__(self, dt):
        self.n = 10
        self.m = 4

        self.mass = 2.02
        self.Ixx = 0.011
        self.Iyy = 0.015
        self.Izz = 0.021
        self.gravity = 9.81
        self.l = 0.26
        self.ct = 5.84e-6
        self.cq = 0.06*self.ct

        self.c1 = (self.Iyy - self.Izz)/self.Ixx
        self.c2 = (self.Izz - self.Ixx)/self.Iyy
        self.c3 = (self.Iyy - self.Ixx)/self.Izz

        self.hover_thrust = self.mass*self.gravity
        self.max_thrust = 28.3
        self.min_thrust = 0.0

        self.hover_w2 = (self.hover_thrust)/(4*self.ct)
        self.max_w2 = (self.max_thrust)/(4*self.ct)
        self.max_tx = self.ct*self.l*self.max_w2
        self.max_ty = self.ct*self.l*self.max_w2
        self.max_tz = 2*self.cq*self.max_w2

        self.dt = dt                    # time between steps in seconds
        self.N = 15                     # number of look ahead steps

        # state symbolic variables
        self.x = ca.SX.sym('x')
        self.y = ca.SX.sym('y')
        self.z = ca.SX.sym('z')
        self.q_x = ca.SX.sym('q_x')
        self.q_y = ca.SX.sym('q_y')
        self.q_z = ca.SX.sym('q_z')
        self.q_w = ca.SX.sym('q_w')
        self.dx = ca.SX.sym('dx')
        self.dy = ca.SX.sym('dy')
        self.dz = ca.SX.sym('dz')
        self.states = ca.vertcat(
            self.x,
            self.y,
            self.z,
            self.q_x,
            self.q_y,
            self.q_z,
            self.q_w,
            self.dx,
            self.dy,
            self.dz
        )
        self.n_states = self.states.numel()

        # control symbolic variables
        self.Th = ca.SX.sym('Th')
        self.omega_x = ca.SX.sym('omega_x')
        self.omega_y = ca.SX.sym('omega_y')
        self.omega_z = ca.SX.sym('omega_z')
        self.controls = ca.vertcat(
            self.Th,
            self.omega_x,
            self.omega_y,
            self.omega_z
        )
        self.n_controls = self.controls.numel()

        # matrix containing all states over all time steps +1 (each column is a state vector)
        self.X = ca.SX.sym('X', self.n_states, self.N + 1)

        # matrix containing all control actions over all time steps (each column is an action vector)
        self.U = ca.SX.sym('U', self.n_controls, self.N)

        # coloumn vector for storing initial state and target state
        self.P = ca.SX.sym('P', self.n_states + self.n_states + self.n_controls)

        # state weights matrix
        self.Q = ca.diagcat(10.0, 10.0, 10.0, 
                    10.0, 10.0, 10.0, 
                    0.2, 0.2, 0.2)
        
        #self.QN = self.Q
        self.QN = ca.diagcat(20.0, 20.0, 20.0, 
                    20.0, 20.0, 20.0, 
                    1.0, 1.0, 1.0)

        # controls weights matrix
        #self.R = ca.diagcat(5e-2, 5e-2, 5e-2, 5e-2)
        #self.R = ca.diagcat(1e-1, 1e-1, 1e-1, 1e-1)
        self.R = ca.diagcat(1.0, 1.0, 1.0, 1.0)
        #self.R = ca.diagcat(0.2, 0.2, 0.2, 0.2)

        #self.RN = self.R
        self.RN = ca.diagcat(0.0, 0.0, 0.0, 0.0)

        # discretization model
        self.quad_RHS = ca.vertcat(
            self.dx,
            self.dy,
            self.dz,
            0.5*(self.omega_x*self.q_w - self.omega_y*self.q_z + self.omega_z*self.q_y),
            0.5*(self.omega_x*self.q_z + self.omega_y*self.q_w - self.omega_z*self.q_x),
            0.5*(-self.omega_x*self.q_y + self.omega_y*self.q_x + self.omega_z*self.q_w),
            0.5*(-self.omega_x*self.q_x - self.omega_y*self.q_y - self.omega_z*self.q_z),
            2*(self.q_x*self.q_z + self.q_w*self.q_y)*((self.Th*0.1*self.max_thrust)/self.mass),
            2*(self.q_y*self.q_z - self.q_w*self.q_x)*((self.Th*0.1*self.max_thrust)/self.mass),
            (1 - 2*(self.q_x**2 + self.q_y**2))*((self.Th*0.1*self.max_thrust)/self.mass) - self.gravity
        )

        # maps controls from [va, vb, vc, vd].T to [vx, vy, omega].T
        self.f = ca.Function('f', [self.states, self.controls], [self.quad_RHS])

        self.cost_fn = 0  # cost function
        self.g = self.X[:, 0] - self.P[:self.n_states]  # constraints in the equation

        # euler
        for k in range(self.N):
            st = self.X[:, k]
            con = self.U[:, k]
            if k == self.N - 1:
                Q = self.QN
            else:
                Q = self.Q
            P = self.P
            if k == self.N - 1:
                R = self.RN
            else:
                R = self.R
            self.cost_fn = self.cost_fn \
                + (st[:3] - P[self.n_states:self.n_states+3]).T @ self.Q[:3,:3] @ (st[:3] - P[self.n_states:self.n_states+3]) \
                + Q[3,3]*(st[3]*P[16] - st[4]*P[15] + st[5]*P[14] - st[6]*P[13]).T @ (st[3]*P[16] - st[4]*P[15] + st[5]*P[14] - st[6]*P[13]) \
                + Q[4,4]*(st[3]*P[15] + st[4]*P[16] - st[5]*P[13] - st[6]*P[13]).T @ (st[3]*P[15] + st[4]*P[16] - st[5]*P[13] - st[6]*P[13]) \
                + Q[5,5]*(-st[3]*P[14] + st[4]*P[13] + st[5]*P[16] - st[6]*P[15]).T @ (-st[3]*P[14] + st[4]*P[13] + st[5]*P[16] - st[6]*P[15]) \
                + (st[7:] - P[self.n_states+7:self.n_states+10]).T @ Q[6:,6:] @ (st[7:] - P[self.n_states+7:self.n_states+10]) \
                + (con - P[self.n_states+10:]).T @ R @ (con - P[self.n_states+10:])
            st_next = self.X[:, k+1]
            k1 = self.f(st, con)
            st_next_RK4 = st + (self.dt)*(k1)
            self.g = ca.vertcat(self.g, st_next - st_next_RK4)


        self.OPT_variables = ca.vertcat(
            self.X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
            self.U.reshape((-1, 1))
        )
        self.nlp_prob = {
            'f': self.cost_fn,
            'x': self.OPT_variables,
            'g': self.g,
            'p': P
        }

        self.opts = {
            'ipopt': {
                'max_iter': 2000,
                'print_level': 0,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6,
                'linear_solver': 'MA27'
            },
            'print_time': 0
        }

        self.solver = ca.nlpsol('solver', 'ipopt', self.nlp_prob, self.opts)

        self.lbx = ca.DM.zeros((self.n_states*(self.N+1) + self.n_controls*self.N, 1))
        self.ubx = ca.DM.zeros((self.n_states*(self.N+1) + self.n_controls*self.N, 1))

        self.lbx[0: self.n_states*(self.N+1): self.n_states] = -ca.inf      # x lower bound
        self.lbx[1: self.n_states*(self.N+1): self.n_states] = -ca.inf      # y lower bound
        self.lbx[2: self.n_states*(self.N+1): self.n_states] = -ca.inf      # z lower bound
        self.lbx[3: self.n_states*(self.N+1): self.n_states] = -ca.inf      # q_x lower bound
        self.lbx[4: self.n_states*(self.N+1): self.n_states] = -ca.inf      # q_y lower bound
        self.lbx[5: self.n_states*(self.N+1): self.n_states] = -ca.inf      # q_z lower bound
        self.lbx[6: self.n_states*(self.N+1): self.n_states] = -ca.inf      # q_w lower bound
        self.lbx[7: self.n_states*(self.N+1): self.n_states] = -ca.inf      # dx lower bound
        self.lbx[8: self.n_states*(self.N+1): self.n_states] = -ca.inf      # dy lower bound
        self.lbx[9: self.n_states*(self.N+1): self.n_states] = -ca.inf      # dz lower bound

        self.ubx[0: self.n_states*(self.N+1): self.n_states] = ca.inf      # x upper bound
        self.ubx[1: self.n_states*(self.N+1): self.n_states] = ca.inf      # y upper bound
        self.ubx[2: self.n_states*(self.N+1): self.n_states] = ca.inf      # z upper bound
        self.ubx[3: self.n_states*(self.N+1): self.n_states] = ca.inf      # q_x upper bound
        self.ubx[4: self.n_states*(self.N+1): self.n_states] = ca.inf      # q_y upper bound
        self.ubx[5: self.n_states*(self.N+1): self.n_states] = ca.inf      # q_z upper bound
        self.ubx[6: self.n_states*(self.N+1): self.n_states] = ca.inf      # q_w upper bound
        self.ubx[7: self.n_states*(self.N+1): self.n_states] = ca.inf      # dx upper bound
        self.ubx[8: self.n_states*(self.N+1): self.n_states] = ca.inf      # dy upper bound
        self.ubx[9: self.n_states*(self.N+1): self.n_states] = ca.inf      # dz upper bound

        self.lbx[self.n_states*(self.N+1): self.n_states*(self.N+1) + self.n_controls*self.N: self.n_controls] = 0.0                # Th lower bound
        self.lbx[self.n_states*(self.N+1) + 1: self.n_states*(self.N+1) + self.n_controls*self.N: self.n_controls] = -2.0           # omega_x lower bound
        self.lbx[self.n_states*(self.N+1) + 2: self.n_states*(self.N+1) + self.n_controls*self.N: self.n_controls] = -2.0           # omega_y lower bound
        self.lbx[self.n_states*(self.N+1) + 3: self.n_states*(self.N+1) + self.n_controls*self.N: self.n_controls] = -2.0           # omega_z lower bound

        self.ubx[self.n_states*(self.N+1): self.n_states*(self.N+1) + self.n_controls*self.N: self.n_controls] = 10.0               # Th upper bound
        self.ubx[self.n_states*(self.N+1) + 1: self.n_states*(self.N+1) + self.n_controls*self.N: self.n_controls] = 2.0            # omega_x upper bound
        self.ubx[self.n_states*(self.N+1) + 2: self.n_states*(self.N+1) + self.n_controls*self.N: self.n_controls] = 2.0            # omega_y upper bound
        self.ubx[self.n_states*(self.N+1) + 3: self.n_states*(self.N+1) + self.n_controls*self.N: self.n_controls] = 2.0            # omega_z upper bound


        self.args = {
            'lbg': ca.DM.zeros((self.n_states*(self.N+1), 1)),  # constraints lower bound
            'ubg': ca.DM.zeros((self.n_states*(self.N+1), 1)),  # constraints upper bound
            'lbx': self.lbx,
            'ubx': self.ubx
        }

    def simulate(self, step_horizon, t0, state_init, u):

        u_ctrl = np.squeeze(u[:,1])

        def lin_func(t, x):
            xdot = [x[7], x[8], x[9],
                    0.5*(u_ctrl[1]*x[6] - u_ctrl[2]*x[5] + u_ctrl[3]*x[4]),
                    0.5*(u_ctrl[1]*x[5] + u_ctrl[2]*x[6] - u_ctrl[3]*x[3]),
                    0.5*(-u_ctrl[1]*x[4] + u_ctrl[2]*x[3] + u_ctrl[3]*x[6]),
                    0.5*(-u_ctrl[1]*x[3] - u_ctrl[2]*x[4] - u_ctrl[3]*x[5]),
                    2*(x[3]*x[5] + x[6]*x[4])*((u_ctrl[0]*0.1*self.max_thrust)/self.mass),
                    2*(x[4]*x[5] - x[6]*x[3])*((u_ctrl[0]*0.1*self.max_thrust)/self.mass),
                    (1 - 2*(x[3]**2 + x[4]**2))*((u_ctrl[0]*0.1*self.max_thrust)/self.mass) - self.gravity]
            return xdot
        soln = solve_ivp(lin_func, [0.0, step_horizon], np.squeeze(state_init), t_eval = [step_horizon])
        xnext = soln.y
        next_state =  np.squeeze(xnext)

        return next_state
    
    def DM2Arr(self, dm):
        return np.array(dm.full())
    
    def ctrl_loop(self, state_init, state_target, ctrl_init):

        # Control Setup

        t0 = 0
        state_init = ca.DM(state_init)          # initial state
        state_target = ca.DM(state_target)      # target state

        # xx = DM(state_init)
        t = ca.DM(t0)

        u0 = ca.repmat(ctrl_init, 1, self.N)            # initial control
        X0 = ca.repmat(state_init, 1, self.N+1)         # initial state full


        mpc_iter = 0
        cat_states = self.DM2Arr(X0)
        cat_controls = self.DM2Arr(u0[:, 0])
        times = np.array([[0]])

        # Control Loop

        self.args['p'] = ca.vertcat(
            state_init,     # current state
            state_target,   # target state
            ctrl_init       # reference control
        )
        # optimization variable current state
        self.args['x0'] = ca.vertcat(
            ca.reshape(X0, self.n_states*(self.N+1), 1),
            ca.reshape(u0, self.n_controls*self.N, 1)
        )

        sol = self.solver(
            x0=self.args['x0'],
            lbx=self.args['lbx'],
            ubx=self.args['ubx'],
            lbg=self.args['lbg'],
            ubg=self.args['ubg'],
            p=self.args['p']
        )

        u = ca.reshape(sol['x'][self.n_states * (self.N + 1):], self.n_controls, self.N)
        X0 = ca.reshape(sol['x'][: self.n_states * (self.N+1)], self.n_states, self.N+1)

        cat_states = np.dstack((
            cat_states,
            self.DM2Arr(X0)
        ))

        cat_controls = np.vstack((
            cat_controls,
            self.DM2Arr(u[:, 0])
        ))
        t = np.vstack((
            t,
            t0
        ))

        state_new = self.simulate(self.dt, t0, state_init, u)

        return np.squeeze(u[:,1]), np.squeeze(state_new)[3:7]



