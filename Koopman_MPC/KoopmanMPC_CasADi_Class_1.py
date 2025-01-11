from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, pi
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as Rot

from scipy.integrate import solve_ivp


class Quad_KoopmanMPC:
    def __init__(self, dt, KoopMat, basis, quadspec, u_ulim, u_llim, xscal, uscal):
        self.n = 10
        self.m = 4

        self.nz = 23

        self.mass = quadspec[0]
        self.Ixx = quadspec[1]
        self.Iyy = quadspec[2]
        self.Izz = quadspec[3]
        self.gravity = 9.81
        self.l = quadspec[4]
        self.ct = quadspec[5]
        self.cq = quadspec[6]

        self.c1 = (self.Iyy - self.Izz)/self.Ixx
        self.c2 = (self.Izz - self.Ixx)/self.Iyy
        self.c3 = (self.Iyy - self.Ixx)/self.Izz

        self.hover_thrust = self.mass*self.gravity
        self.max_thrust = quadspec[7]
        self.min_thrust = 0.0

        self.hover_w2 = (self.hover_thrust)/(4*self.ct)
        self.max_w2 = (self.max_thrust)/(4*self.ct)
        self.max_tx = self.ct*self.l*self.max_w2
        self.max_ty = self.ct*self.l*self.max_w2
        self.max_tz = 2*self.cq*self.max_w2

        self.dt = dt                    # time between steps in seconds
        self.N = 25                     # number of look ahead steps
        #self.N = 10                     # number of look ahead steps

        self.xscal = xscal
        self.uscal = uscal

        self.Amat = KoopMat[0,:,:]
        self.B1mat = KoopMat[1,:,:]
        self.B2mat = KoopMat[2,:,:]
        self.B3mat = KoopMat[3,:,:]
        self.B4mat = KoopMat[4,:,:]

        self.basis = basis

        # state symbolic variables
        self.z = ca.SX.sym('z', self.nz)
        self.states = ca.vertcat(
            self.z
        )
        #self.n_states = self.states.numel()
        self.n_states = self.nz

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
        #self.Q = ca.diagcat(10.0, 10.0, 10.0, 
        #            1.0, 1.0, 1.0, 
        #            0.01, 0.01, 0.01)

        #self.Q = ca.diagcat(10.0, 10.0, 10.0, 
        #            1.0, 1.0, 1.0, 
        #            1.0, 1.0, 1.0)
        
        #self.Q = ca.diagcat(20.0, 20.0, 20.0, 
        #            20.0, 20.0, 20.0, 
        #            0.05, 0.05, 0.05)

        self.Q = ca.diagcat(20.0, 20.0, 20.0, 
                    10.0, 10.0, 10.0, 
                    0.02, 0.02, 0.02)
        
        self.QN = self.Q
        #self.QN = ca.diagcat(20.0, 20.0, 20.0, 
        #            20.0, 20.0, 20.0, 
        #            0.5, 0.5, 0.5)

        # controls weights matrix
        #self.R = ca.diagcat(5e-2, 5e-2, 5e-2, 5e-2)
        #self.R = ca.diagcat(1e-1, 1e-1, 1e-1, 1e-1)
        #self.R = ca.diagcat(1.0, 1.0, 1.0, 1.0)
        self.R = ca.diagcat(0.5, 0.5, 0.5, 0.5)

        self.RN = self.R
        #self.RN = ca.diagcat(0.0, 0.0, 0.0, 0.0)

        # discretization model
        self.quad_RHS = (self.Amat + self.B1mat*self.Th + self.B2mat*self.omega_x + self.B3mat*self.omega_y + self.B4mat*self.omega_z)@self.z

        # maps controls from [va, vb, vc, vd].T to [vx, vy, omega].T
        self.f = ca.Function('f', [self.states, self.controls], [self.quad_RHS])

        self.cost_fn = 0  # cost function
        self.g = self.X[:, 0] - self.P[:self.n_states]  # constraints in the equation

        '''
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
                + (st[1:4] - P[self.n_states+1:self.n_states+4]).T @ self.Q[:3,:3] @ (st[1:4] - P[self.n_states+1:self.n_states+4]) \
                + Q[3,3]*(st[4]*P[30] - st[5]*P[29] + st[6]*P[28] - st[7]*P[27]).T @ (st[4]*P[30] - st[5]*P[29] + st[6]*P[28] - st[7]*P[27]) \
                + Q[4,4]*(st[4]*P[29] + st[5]*P[30] - st[6]*P[27] - st[7]*P[27]).T @ (st[4]*P[29] + st[5]*P[30] - st[6]*P[27] - st[7]*P[27]) \
                + Q[5,5]*(-st[4]*P[28] + st[5]*P[27] + st[6]*P[30] - st[7]*P[29]).T @ (-st[4]*P[28] + st[5]*P[27] + st[6]*P[30] - st[7]*P[29]) \
                + (st[8:11] - P[self.n_states+8:self.n_states+11]).T @ Q[6:,6:] @ (st[8:11] - P[self.n_states+8:self.n_states+11]) \
                + (con - P[self.n_states+23:]).T @ R @ (con - P[self.n_states+23:])
            st_next = self.X[:, k+1]
            st_next_koop = st + (self.dt)*(self.f(st, con))
            self.g = ca.vertcat(self.g, st_next - st_next_koop)
        '''

        # euler
        for k in range(self.N):
            P = self.P

            st_z = self.X[:, k]
            st = (self.X[1:11, k]*self.xscal[:,0]) + self.xscal[:,1]
            st_targ = (P[self.n_states+1:self.n_states+11]*self.xscal[:,0]) + self.xscal[:,1]

            con_norm = self.U[:, k]
            con = (self.U[:, k]*self.uscal[:,0]) + self.uscal[:,1]
            con_targ = (P[self.n_states+23:]*self.uscal[:,0]) + self.uscal[:,1]

            if k == self.N - 1:
                Q = self.QN
            else:
                Q = self.Q
            if k == self.N - 1:
                R = self.RN
            else:
                R = self.R
            self.cost_fn = self.cost_fn \
                + (st[:3] - st_targ[:3]).T @ self.Q[:3,:3] @ (st[:3] - st_targ[:3]) \
                + Q[3,3]*(st[3]*st_targ[6] - st[4]*st_targ[5] + st[5]*st_targ[4] - st[6]*st_targ[3]).T @ (st[3]*st_targ[6] - st[4]*st_targ[5] + st[5]*st_targ[4] - st[6]*st_targ[3]) \
                + Q[4,4]*(st[3]*st_targ[5] + st[4]*st_targ[6] - st[5]*st_targ[3] - st[6]*st_targ[3]).T @ (st[3]*st_targ[5] + st[4]*st_targ[6] - st[5]*st_targ[3] - st[6]*st_targ[3]) \
                + Q[5,5]*(-st[3]*st_targ[4] + st[4]*st_targ[3] + st[5]*st_targ[6] - st[6]*st_targ[5]).T @ (-st[3]*st_targ[4] + st[4]*st_targ[3] + st[5]*st_targ[6] - st[6]*st_targ[5]) \
                + (st[7:] - st_targ[7:]).T @ Q[6:,6:] @ (st[7:] - st_targ[7:]) \
                + (con - con_targ).T @ R @ (con - con_targ)
            st_next = self.X[:, k+1]
            st_next_koop = st_z + (self.dt)*(self.f(st_z, con_norm))
            self.g = ca.vertcat(self.g, st_next - st_next_koop)

            #st_next_x = (st_next[1:11]*self.xscal[:,0]) + self.xscal[:,1]
            #st_next_quat = st_next_x[3:7]
            #self.g = ca.vertcat(self.g, np.sqrt((st_next_quat[0]**2) + (st_next_quat[1]**2) + (st_next_quat[2]**2) + (st_next_quat[3]**2)) - 1)

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

        '''
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
        '''

        self.lbx[:self.n_states*(self.N+1)] = -ca.inf
        self.ubx[:self.n_states*(self.N+1)] = ca.inf

        self.lbx[self.n_states*(self.N+1): self.n_states*(self.N+1) + self.n_controls*self.N: self.n_controls] = u_llim[0]               # Th lower bound
        self.lbx[self.n_states*(self.N+1) + 1: self.n_states*(self.N+1) + self.n_controls*self.N: self.n_controls] = u_llim[1]           # omega_x lower bound
        self.lbx[self.n_states*(self.N+1) + 2: self.n_states*(self.N+1) + self.n_controls*self.N: self.n_controls] = u_llim[2]           # omega_y lower bound
        self.lbx[self.n_states*(self.N+1) + 3: self.n_states*(self.N+1) + self.n_controls*self.N: self.n_controls] = u_llim[3]           # omega_z lower bound

        self.ubx[self.n_states*(self.N+1): self.n_states*(self.N+1) + self.n_controls*self.N: self.n_controls] = u_ulim[0]               # Th upper bound
        self.ubx[self.n_states*(self.N+1) + 1: self.n_states*(self.N+1) + self.n_controls*self.N: self.n_controls] = u_ulim[1]           # omega_x upper bound
        self.ubx[self.n_states*(self.N+1) + 2: self.n_states*(self.N+1) + self.n_controls*self.N: self.n_controls] = u_ulim[2]           # omega_y upper bound
        self.ubx[self.n_states*(self.N+1) + 3: self.n_states*(self.N+1) + self.n_controls*self.N: self.n_controls] = u_ulim[3]           # omega_z upper bound


        self.args = {
            'lbg': ca.DM.zeros((self.n_states*(self.N+1), 1)),  # constraints lower bound
            'ubg': ca.DM.zeros((self.n_states*(self.N+1), 1)),  # constraints upper bound
            'lbx': self.lbx,
            'ubx': self.ubx
        }

        #self.args = {
        #    'lbg': ca.DM.zeros((self.n_states*(self.N+1) + self.N, 1)),  # constraints lower bound
        #    'ubg': ca.DM.zeros((self.n_states*(self.N+1) + self.N, 1)),  # constraints upper bound
        #    'lbx': self.lbx,
        #    'ubx': self.ubx
        #}

    def simulate(self, step_horizon, t0, state_init, u):

        u_ctrl = np.squeeze(u[:,1])

        u_ctrl = (u_ctrl*self.uscal[:,0]) + self.uscal[:,1]

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
    
    def ctrl_loop(self, state_init_x, state_target_x, ctrl_init):

        # Control Setup

        t0 = 0
        state_init = ca.DM(self.basis(state_init_x))          # initial state
        state_target = ca.DM(self.basis(state_target_x))      # target state

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

        state_new = self.simulate(self.dt, t0, state_init_x, u)

        return np.squeeze(u[:,1]), np.squeeze(state_new)[3:7]
    
    def ctrl_loop_debug(self, state_init_x, state_target_x, ctrl_init):

        # Control Setup

        t0 = 0
        state_init = ca.DM(self.basis(state_init_x))          # initial state
        state_target = ca.DM(self.basis(state_target_x))      # target state

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

        return self.DM2Arr(u), self.DM2Arr(X0)
    
class Quad_KoopmanMPC_Euler:
    def __init__(self, dt, KoopMat, basis, quadspec, u_ulim, u_llim, xscal, uscal):
        self.n = 9
        self.m = 4

        self.nz = 23

        self.mass = quadspec[0]
        self.Ixx = quadspec[1]
        self.Iyy = quadspec[2]
        self.Izz = quadspec[3]
        self.gravity = 9.81
        self.l = quadspec[4]
        self.ct = quadspec[5]
        self.cq = quadspec[6]

        self.c1 = (self.Iyy - self.Izz)/self.Ixx
        self.c2 = (self.Izz - self.Ixx)/self.Iyy
        self.c3 = (self.Iyy - self.Ixx)/self.Izz

        self.hover_thrust = self.mass*self.gravity
        self.max_thrust = quadspec[7]
        self.min_thrust = 0.0

        self.hover_w2 = (self.hover_thrust)/(4*self.ct)
        self.max_w2 = (self.max_thrust)/(4*self.ct)
        self.max_tx = self.ct*self.l*self.max_w2
        self.max_ty = self.ct*self.l*self.max_w2
        self.max_tz = 2*self.cq*self.max_w2

        self.dt = dt                    # time between steps in seconds
        self.N = 25                     # number of look ahead steps
        #self.N = 10                     # number of look ahead steps

        self.Amat = KoopMat[0,:,:]
        self.B1mat = KoopMat[1,:,:]
        self.B2mat = KoopMat[2,:,:]
        self.B3mat = KoopMat[3,:,:]
        self.B4mat = KoopMat[4,:,:]

        self.basis = basis

        # state symbolic variables
        self.z = ca.SX.sym('z', self.nz)
        self.states = ca.vertcat(
            self.z
        )
        #self.n_states = self.states.numel()
        self.n_states = self.nz

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
        #self.Q = ca.diagcat(10.0, 10.0, 10.0, 
        #            1.0, 1.0, 1.0, 
        #            0.01, 0.01, 0.01)

        self.Q = ca.diagcat(10.0, 10.0, 10.0, 
                    1.0, 1.0, 1.0, 
                    1.0, 1.0, 1.0)
        
        self.QN = self.Q
        #self.QN = ca.diagcat(20.0, 20.0, 20.0, 
        #            20.0, 20.0, 20.0, 
        #            0.5, 0.5, 0.5)

        # controls weights matrix
        #self.R = ca.diagcat(5e-2, 5e-2, 5e-2, 5e-2)
        #self.R = ca.diagcat(1e-1, 1e-1, 1e-1, 1e-1)
        #self.R = ca.diagcat(5e-5, 5e-5, 5e-5, 5e-5)
        #self.R = ca.diagcat(20.0, 20.0, 20.0, 20.0)

        self.R = ca.diagcat(1.0, 1.0, 1.0, 1.0)

        self.RN = self.R
        #self.RN = ca.diagcat(0.0, 0.0, 0.0, 0.0)

        # discretization model
        self.quad_RHS = (self.Amat + self.B1mat*self.Th + self.B2mat*self.omega_x + self.B3mat*self.omega_y + self.B4mat*self.omega_z)@self.z

        # maps controls from [va, vb, vc, vd].T to [vx, vy, omega].T
        self.f = ca.Function('f', [self.states, self.controls], [self.quad_RHS])

        self.cost_fn = 0  # cost function
        self.g = self.X[:, 0] - self.P[:self.n_states]  # constraints in the equation

        '''
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
                + (st[1:10] - P[self.n_states+1:self.n_states+10]).T @ Q @ (st[1:10] - P[self.n_states+1:self.n_states+10]) \
                + (con - P[self.n_states+23:]).T @ R @ (con - P[self.n_states+23:])
            st_next = self.X[:, k+1]
            st_next_koop = st + (self.dt)*(self.f(st, con))
            self.g = ca.vertcat(self.g, st_next - st_next_koop)
        '''

        # euler
        for k in range(self.N):
            P = self.P

            st_z = self.X[:, k]
            st = (self.X[1:10, k]*xscal[:,0]) + xscal[:,1]
            st_targ = (P[self.n_states+1:self.n_states+10]*xscal[:,0]) + xscal[:,1]

            con_norm = self.U[:, k]
            con = (self.U[:, k]*uscal[:,0]) + uscal[:,1]
            con_targ = (P[self.n_states+23:]*uscal[:,0]) + uscal[:,1]

            if k == self.N - 1:
                Q = self.QN
            else:
                Q = self.Q
            if k == self.N - 1:
                R = self.RN
            else:
                R = self.R
            self.cost_fn = self.cost_fn \
                + (st - st_targ).T @ self.Q @ (st - st_targ) \
                + (con - con_targ).T @ R @ (con - con_targ)
            #self.cost_fn = self.cost_fn \
            #    + (st - st_targ).T @ self.Q @ (st - st_targ) \
            #    + con.T @ R @ con
            st_next = self.X[:, k+1]
            st_next_koop = st_z + (self.dt)*(self.f(st_z, con_norm))
            self.g = ca.vertcat(self.g, st_next - st_next_koop)


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

        '''
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
        '''

        self.lbx[:self.n_states*(self.N+1)] = -ca.inf
        self.ubx[:self.n_states*(self.N+1)] = ca.inf

        self.lbx[self.n_states*(self.N+1): self.n_states*(self.N+1) + self.n_controls*self.N: self.n_controls] = u_llim[0]               # Th lower bound
        self.lbx[self.n_states*(self.N+1) + 1: self.n_states*(self.N+1) + self.n_controls*self.N: self.n_controls] = u_llim[1]           # omega_x lower bound
        self.lbx[self.n_states*(self.N+1) + 2: self.n_states*(self.N+1) + self.n_controls*self.N: self.n_controls] = u_llim[2]           # omega_y lower bound
        self.lbx[self.n_states*(self.N+1) + 3: self.n_states*(self.N+1) + self.n_controls*self.N: self.n_controls] = u_llim[3]           # omega_z lower bound

        self.ubx[self.n_states*(self.N+1): self.n_states*(self.N+1) + self.n_controls*self.N: self.n_controls] = u_ulim[0]               # Th upper bound
        self.ubx[self.n_states*(self.N+1) + 1: self.n_states*(self.N+1) + self.n_controls*self.N: self.n_controls] = u_ulim[1]           # omega_x upper bound
        self.ubx[self.n_states*(self.N+1) + 2: self.n_states*(self.N+1) + self.n_controls*self.N: self.n_controls] = u_ulim[2]           # omega_y upper bound
        self.ubx[self.n_states*(self.N+1) + 3: self.n_states*(self.N+1) + self.n_controls*self.N: self.n_controls] = u_ulim[3]           # omega_z upper bound


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
    
    def ctrl_loop(self, state_init_x, state_target_x, ctrl_init):

        # Control Setup

        t0 = 0
        state_init = ca.DM(self.basis(state_init_x))          # initial state
        state_target = ca.DM(self.basis(state_target_x))      # target state

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

        return np.squeeze(u[:,1])
    
    def ctrl_loop_debug(self, state_init_x, state_target_x, ctrl_init):

        # Control Setup

        t0 = 0
        state_init = ca.DM(self.basis(state_init_x))          # initial state
        state_target = ca.DM(self.basis(state_target_x))      # target state

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

        return self.DM2Arr(u), self.DM2Arr(X0)
    
class Quad_KoopmanMPC_RotVec:
    def __init__(self, dt, KoopMat, basis, quadspec, u_ulim, u_llim, xscal, uscal):
        self.n = 9
        self.m = 4

        self.nz = 23

        self.mass = quadspec[0]
        self.Ixx = quadspec[1]
        self.Iyy = quadspec[2]
        self.Izz = quadspec[3]
        self.gravity = 9.81
        self.l = quadspec[4]
        self.ct = quadspec[5]
        self.cq = quadspec[6]

        self.c1 = (self.Iyy - self.Izz)/self.Ixx
        self.c2 = (self.Izz - self.Ixx)/self.Iyy
        self.c3 = (self.Iyy - self.Ixx)/self.Izz

        self.hover_thrust = self.mass*self.gravity
        self.max_thrust = quadspec[7]
        self.min_thrust = 0.0

        self.hover_w2 = (self.hover_thrust)/(4*self.ct)
        self.max_w2 = (self.max_thrust)/(4*self.ct)
        self.max_tx = self.ct*self.l*self.max_w2
        self.max_ty = self.ct*self.l*self.max_w2
        self.max_tz = 2*self.cq*self.max_w2

        self.dt = dt                    # time between steps in seconds
        self.N = 25                     # number of look ahead steps
        #self.N = 10                     # number of look ahead steps

        self.Amat = KoopMat[0,:,:]
        self.B1mat = KoopMat[1,:,:]
        self.B2mat = KoopMat[2,:,:]
        self.B3mat = KoopMat[3,:,:]
        self.B4mat = KoopMat[4,:,:]

        self.basis = basis

        # state symbolic variables
        self.z = ca.SX.sym('z', self.nz)
        self.states = ca.vertcat(
            self.z
        )
        #self.n_states = self.states.numel()
        self.n_states = self.nz

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
        #self.Q = ca.diagcat(10.0, 10.0, 10.0, 
        #            1.0, 1.0, 1.0, 
        #            0.01, 0.01, 0.01)

        self.Q = ca.diagcat(10.0, 10.0, 10.0, 
                    1.0, 1.0, 1.0, 
                    1.0, 1.0, 1.0)
        
        self.QN = self.Q
        #self.QN = ca.diagcat(20.0, 20.0, 20.0, 
        #            20.0, 20.0, 20.0, 
        #            0.5, 0.5, 0.5)

        # controls weights matrix
        #self.R = ca.diagcat(5e-2, 5e-2, 5e-2, 5e-2)
        #self.R = ca.diagcat(1e-1, 1e-1, 1e-1, 1e-1)
        #self.R = ca.diagcat(5e-5, 5e-5, 5e-5, 5e-5)
        #self.R = ca.diagcat(20.0, 20.0, 20.0, 20.0)

        self.R = ca.diagcat(1.0, 1.0, 1.0, 1.0)

        self.RN = self.R
        #self.RN = ca.diagcat(0.0, 0.0, 0.0, 0.0)

        # discretization model
        self.quad_RHS = (self.Amat + self.B1mat*self.Th + self.B2mat*self.omega_x + self.B3mat*self.omega_y + self.B4mat*self.omega_z)@self.z

        # maps controls from [va, vb, vc, vd].T to [vx, vy, omega].T
        self.f = ca.Function('f', [self.states, self.controls], [self.quad_RHS])

        self.cost_fn = 0  # cost function
        self.g = self.X[:, 0] - self.P[:self.n_states]  # constraints in the equation

        '''
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
                + (st[1:10] - P[self.n_states+1:self.n_states+10]).T @ Q @ (st[1:10] - P[self.n_states+1:self.n_states+10]) \
                + (con - P[self.n_states+23:]).T @ R @ (con - P[self.n_states+23:])
            st_next = self.X[:, k+1]
            st_next_koop = st + (self.dt)*(self.f(st, con))
            self.g = ca.vertcat(self.g, st_next - st_next_koop)
        '''

        # euler
        for k in range(self.N):
            P = self.P

            st_z = self.X[:, k]
            st = (self.X[1:10, k]*xscal[:,0]) + xscal[:,1]
            st_targ = (P[self.n_states+1:self.n_states+10]*xscal[:,0]) + xscal[:,1]

            con_norm = self.U[:, k]
            con = (self.U[:, k]*uscal[:,0]) + uscal[:,1]
            con_targ = (P[self.n_states+23:]*uscal[:,0]) + uscal[:,1]

            if k == self.N - 1:
                Q = self.QN
            else:
                Q = self.Q
            if k == self.N - 1:
                R = self.RN
            else:
                R = self.R
            self.cost_fn = self.cost_fn \
                + (st - st_targ).T @ self.Q @ (st - st_targ) \
                + (con - con_targ).T @ R @ (con - con_targ)
            #self.cost_fn = self.cost_fn \
            #    + (st - st_targ).T @ self.Q @ (st - st_targ) \
            #    + con.T @ R @ con
            st_next = self.X[:, k+1]
            st_next_koop = st_z + (self.dt)*(self.f(st_z, con_norm))
            self.g = ca.vertcat(self.g, st_next - st_next_koop)


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

        self.lbx[:self.n_states*(self.N+1)] = -ca.inf
        self.ubx[:self.n_states*(self.N+1)] = ca.inf

        self.lbx[self.n_states*(self.N+1): self.n_states*(self.N+1) + self.n_controls*self.N: self.n_controls] = u_llim[0]               # Th lower bound
        self.lbx[self.n_states*(self.N+1) + 1: self.n_states*(self.N+1) + self.n_controls*self.N: self.n_controls] = u_llim[1]           # omega_x lower bound
        self.lbx[self.n_states*(self.N+1) + 2: self.n_states*(self.N+1) + self.n_controls*self.N: self.n_controls] = u_llim[2]           # omega_y lower bound
        self.lbx[self.n_states*(self.N+1) + 3: self.n_states*(self.N+1) + self.n_controls*self.N: self.n_controls] = u_llim[3]           # omega_z lower bound

        self.ubx[self.n_states*(self.N+1): self.n_states*(self.N+1) + self.n_controls*self.N: self.n_controls] = u_ulim[0]               # Th upper bound
        self.ubx[self.n_states*(self.N+1) + 1: self.n_states*(self.N+1) + self.n_controls*self.N: self.n_controls] = u_ulim[1]           # omega_x upper bound
        self.ubx[self.n_states*(self.N+1) + 2: self.n_states*(self.N+1) + self.n_controls*self.N: self.n_controls] = u_ulim[2]           # omega_y upper bound
        self.ubx[self.n_states*(self.N+1) + 3: self.n_states*(self.N+1) + self.n_controls*self.N: self.n_controls] = u_ulim[3]           # omega_z upper bound


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
    
    def ctrl_loop(self, state_init_x, state_target_x, ctrl_init):

        # Control Setup

        t0 = 0
        state_init = ca.DM(self.basis(state_init_x))          # initial state
        state_target = ca.DM(self.basis(state_target_x))      # target state

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

        return np.squeeze(u[:,1])
    
    def ctrl_loop_debug(self, state_init_x, state_target_x, ctrl_init):

        # Control Setup

        t0 = 0
        state_init = ca.DM(self.basis(state_init_x))          # initial state
        state_target = ca.DM(self.basis(state_target_x))      # target state

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

        return self.DM2Arr(u), self.DM2Arr(X0)



