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

def eultoquat(x_eul, full=0):
    if full == 1:
        x_quat = np.zeros(13)
    else:
        x_quat = np.zeros(10)

    x_quat[:3] = x_eul[:3]
    x_quat[7:] = x_eul[6:]

    x_quat[3:7] = Rot.from_euler('xyz', x_eul[3:6]).as_quat()

    return x_quat

def quattoeul(x_quat, full=0):
    if full == 1:
        x_eul = np.zeros(12)
    else:
        x_eul = np.zeros(9)

    x_eul[:3] = x_quat[:3]
    x_eul[6:] = x_quat[7:]

    x_eul[3:6] = Rot.from_quat(x_quat[3:7]).as_euler('xyz')

    return x_eul

def eultorotvec(x_eul, full=0):
    if full == 1:
        x_rotvec = np.zeros(12)
    else:
        x_rotvec = np.zeros(9)

    x_rotvec[:3] = x_eul[:3]
    x_rotvec[6:] = x_eul[6:]

    x_rotvec[3:6] = Rot.from_euler('xyz', x_eul[3:6]).as_rotvec()

    return x_rotvec

def rotvectoeul(x_rotvec, full=0):
    if full == 1:
        x_eul = np.zeros(12)
    else:
        x_eul = np.zeros(9)

    x_eul[:3] = x_rotvec[:3]
    x_eul[6:] = x_rotvec[6:]

    x_eul[3:6] = Rot.from_rotvec(x_rotvec[3:6]).as_euler('xyz')

    return x_eul

def quattorotvec(x_quat, full=0):
    if full == 1:
        x_rotvec = np.zeros(12)
    else:
        x_rotvec = np.zeros(9)

    x_rotvec[:3] = x_quat[:3]
    x_rotvec[6:] = x_quat[7:]

    x_rotvec[3:6] = Rot.from_quat(x_quat[3:7]).as_rotvec()

    return x_rotvec

def rotvectoquat(x_rotvec, full=0):
    if full == 1:
        x_quat = np.zeros(13)
    else:
        x_quat = np.zeros(10)

    x_quat[:3] = x_rotvec[:3]
    x_quat[7:] = x_rotvec[6:]

    x_quat[3:7] = Rot.from_rotvec(x_rotvec[3:6]).as_quat()

    return x_quat



