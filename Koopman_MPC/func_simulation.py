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

def simulate_koop(state_init, u_ctrl, koopmat, basis, xscal, uscal, dt, n):

    u_ctrl_norm = (u_ctrl-uscal[:,1])/uscal[:,0]

    state_init_norm = (state_init-xscal[:,1])/xscal[:,0]
    state_init_z = basis(state_init_norm)

    next_state_z = state_init_z + dt*(koopmat[0,:,:] + koopmat[1,:,:]*u_ctrl_norm[0] + koopmat[2,:,:]*u_ctrl_norm[1] + koopmat[3,:,:]*u_ctrl_norm[2] + koopmat[4,:,:]*u_ctrl_norm[3])@state_init_z

    next_state_euler = (next_state_z[1:n+1]*xscal[:,0]) + xscal[:,1]

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
                ((cos(x[5])*sin(x[4])*cos(x[3]) + sin(x[5])*sin(x[3]))/mass)*(u_ctrl[0]),
                ((sin(x[5])*sin(x[4])*cos(x[3]) - cos(x[5])*sin(x[3]))/mass)*(u_ctrl[0]),
                -gravity + ((cos(x[4])*cos(x[3]))/mass)*(u_ctrl[0])]
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
                2*(x[3]*x[5] + x[6]*x[4])*((u_ctrl[0])/mass),
                2*(x[4]*x[5] - x[6]*x[3])*((u_ctrl[0])/mass),
                (x[6]**2 - x[3]**2 - x[4]**2 + x[5]**2)*((u_ctrl[0])/mass) - gravity]
        return xdot

    soln = solve_ivp(lin_func, [0.0, step_horizon], np.squeeze(state_init), t_eval = [step_horizon])
    xnext = soln.y
    next_state =  np.squeeze(xnext)

    return next_state, u_ctrl

def simulate_quat_full(step_horizon, t0, state_init, u_ctrl, quadspec):

    mass = quadspec[0]
    Ixx = quadspec[1]
    Iyy = quadspec[2]
    Izz = quadspec[3]
    gravity = 9.81
    l = quadspec[4]
    ct = quadspec[5]
    cq = quadspec[6]
    max_thrust = quadspec[7]

    c1 = (Iyy - Izz)/Ixx
    c2 = (Izz - Ixx)/Iyy
    c3 = (Iyy - Ixx)/Izz

    s = 1/(np.linalg.norm(state_init[3:7]))

    def lin_func(t, x):
        xdot = [x[7], x[8], x[9],
                0.5*(1/(np.linalg.norm(x[3:7])))*(x[10]*x[6] - x[11]*x[5] + x[12]*x[4]),
                0.5*(1/(np.linalg.norm(x[3:7])))*(x[10]*x[5] + x[11]*x[6] - x[12]*x[3]),
                0.5*(1/(np.linalg.norm(x[3:7])))*(-x[10]*x[4] + x[11]*x[3] + x[12]*x[6]),
                0.5*(1/(np.linalg.norm(x[3:7])))*(-x[10]*x[3] - x[11]*x[4] - x[12]*x[5]),
                2*(x[3]*x[5] + x[6]*x[4])*(((ct)*(u_ctrl[0] + u_ctrl[1] + u_ctrl[2] + u_ctrl[3]))/mass),
                2*(x[4]*x[5] - x[6]*x[3])*(((ct)*(u_ctrl[0] + u_ctrl[1] + u_ctrl[2] + u_ctrl[3]))/mass),
                (x[6]**2 - x[3]**2 - x[4]**2 + x[5]**2)*(((ct)*(u_ctrl[0] + u_ctrl[1] + u_ctrl[2] + u_ctrl[3]))/mass) - gravity,
                x[11]*x[12]*c1 + ((ct*l)*((u_ctrl[1]) - (u_ctrl[3])))/Ixx,
                x[10]*x[12]*c2 + ((ct*l)*(- (u_ctrl[0]) + (u_ctrl[2])))/Iyy,
                x[10]*x[11]*c3 + ((cq)*(- (u_ctrl[0]) + (u_ctrl[1]) - (u_ctrl[2]) + (u_ctrl[3])))/Izz]
        return xdot

    soln = solve_ivp(lin_func, [0.0, step_horizon], np.squeeze(state_init), t_eval = [step_horizon])
    xnext = soln.y
    next_state =  np.squeeze(xnext)

    return next_state, u_ctrl



