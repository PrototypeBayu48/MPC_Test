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

def simulate_koop(state_init, u_ctrl, koopmat, basis, xscal, uscal, dt):

    u_ctrl_norm = (u_ctrl-uscal[:,1])/uscal[:,0]

    state_init_norm = (state_init-xscal[:,1])/xscal[:,0]
    state_init_z = basis(state_init_norm)

    next_state_z = state_init_z + dt*(koopmat[0,:,:] + koopmat[1,:,:]*u_ctrl_norm[0] + koopmat[2,:,:]*u_ctrl_norm[1] + koopmat[3,:,:]*u_ctrl_norm[2] + koopmat[4,:,:]*u_ctrl_norm[3])@state_init_z

    next_state_euler = (next_state_z[1:10]*xscal[:,0]) + xscal[:,1]

    return next_state_euler, u_ctrl

def simulate_euler(step_horizon, t0, state_init, u_ctrl, quadspec):

    mass = quadspec[0]
    gravity = 9.81
    max_thrust = quadspec[7]

    def lin_func(t, x):
        xdot = [x[6], x[7], x[8], 
                u_ctrl[1] + u_ctrl[2]*sin(x[3])*tan(x[4]) + u_ctrl[3]*cos(x[3])*tan(x[4]), 
                u_ctrl[2]*cos(x[3]) - u_ctrl[3]*sin(x[3]), 
                u_ctrl[2]*(sin(x[3])/cos(x[4])) + u_ctrl[3]*(cos(x[3])/cos(x[4])), 
                ((cos(x[5])*sin(x[4])*cos(x[3]) + sin(x[5])*sin(x[3]))/mass)*(u_ctrl[0]*0.1*max_thrust),
                ((sin(x[5])*sin(x[4])*cos(x[3]) - cos(x[5])*sin(x[3]))/mass)*(u_ctrl[0]*0.1*max_thrust),
                -gravity + ((cos(x[4])*cos(x[3]))/mass)*(u_ctrl[0]*0.1*max_thrust)]
        return xdot
    
    soln = solve_ivp(lin_func, [0.0, step_horizon], np.squeeze(state_init), t_eval = [step_horizon])
    xnext = soln.y
    next_state =  np.squeeze(xnext)

    return next_state, u_ctrl

def simulate_quat(step_horizon, t0, state_init, u_ctrl, quadspec):

    mass = quadspec[0]
    gravity = 9.81
    max_thrust = quadspec[7]

    s = 1/(np.linalg.norm(state_init[3:7]))

    def lin_func(t, x):
        xdot = [x[7], x[8], x[9],
                0.5*(1/(np.linalg.norm(x[3:7])))*(u_ctrl[1]*x[6] - u_ctrl[2]*x[5] + u_ctrl[3]*x[4]),
                0.5*(1/(np.linalg.norm(x[3:7])))*(u_ctrl[1]*x[5] + u_ctrl[2]*x[6] - u_ctrl[3]*x[3]),
                0.5*(1/(np.linalg.norm(x[3:7])))*(-u_ctrl[1]*x[4] + u_ctrl[2]*x[3] + u_ctrl[3]*x[6]),
                0.5*(1/(np.linalg.norm(x[3:7])))*(-u_ctrl[1]*x[3] - u_ctrl[2]*x[4] - u_ctrl[3]*x[5]),
                2*(x[3]*x[5] + x[6]*x[4])*((u_ctrl[0]*0.1*max_thrust)/mass),
                2*(x[4]*x[5] - x[6]*x[3])*((u_ctrl[0]*0.1*max_thrust)/mass),
                (x[6]**2 - x[3]**2 - x[4]**2 + x[5]**2)*((u_ctrl[0]*0.1*max_thrust)/mass) - gravity]
        return xdot

    soln = solve_ivp(lin_func, [0.0, step_horizon], np.squeeze(state_init), t_eval = [step_horizon])
    xnext = soln.y
    next_state =  np.squeeze(xnext)

    return next_state, u_ctrl



