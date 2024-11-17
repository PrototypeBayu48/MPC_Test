import numpy as np
import scipy as sc
import random as rand
import matplotlib.pyplot as plt
import os

import do_mpc
from numpy import sin, cos, tan
import random as rand

from time import time

import pickle

import pypose as pp

def dataprocess(x_train,w_train):
    u_train = np.empty((n_traj_train, datagen_iter, m))

    for n1 in range(8):
        Th = ct*(((w_train[n1,:,0])) + ((w_train[n1,:,1])) + ((w_train[n1,:,2])) + ((w_train[n1,:,3])))
        tx = ct*l*(((w_train[n1,:,1])) - ((w_train[n1,:,3])))
        ty = ct*l*(- ((w_train[n1,:,0])) + ((w_train[n1,:,2])))
        tz = cq*( - ((w_train[n1,:,0])) + ((w_train[n1,:,1])) - ((w_train[n1,:,2])) + ((w_train[n1,:,3])))
        u_train[n1,:,0] = Th
        u_train[n1,:,1] = tx
        u_train[n1,:,2] = ty
        u_train[n1,:,3] = tz

    u_train[:,:,0] = ((u_train[:,:,0])/(0.1*max_thrust))

    u_train[:,:,1] = (u_train[:,:,1])/(0.2*max_tx)

    u_train[:,:,2] = (u_train[:,:,2])/(0.2*max_ty)

    u_train[:,:,3] = (u_train[:,:,3])/(0.2*max_tz)

    x_train[:,:,3:6] = pp.so3(x_train[:,:,3:6]).tensor()

    return x_train,u_train

n = 12
m = 4
dt = 0.02

blo = 1
discrete = 0

koopnet = 0
koopnet_train = 0

groundeffect = 0

mass = 0.0336
Ixx = 1.7e-5
Iyy = 1.7e-5
Izz = 2.9e-5
gravity = 9.81
l = 0.04
ct = 3.1582e-10
cq = 7.9379e-12

R = (45e-3)/2
vh = np.sqrt(((mass*gravity)/4)/(2*1.225*(np.pi)*(R**2)))

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

hover_w2 = (mass*gravity)/(4*ct)
max_w2 = (0.057*gravity)/(4*ct)
hover_thrust = (mass*gravity)
max_thrust = (0.057*gravity)
max_tx = ct*l*max_w2
max_ty = ct*l*max_w2
max_tz = 2*cq*max_w2
uhover = np.array([[hover_thrust], [0.0], [0.0], [0.0]])
uhover_norm = uhover/(0.1*max_thrust)

print(hover_thrust)
print(max_thrust)
print(uhover)
print(uhover_norm)

n_traj_train = 8
n_traj_val = int(0.25*n_traj_train)

datagen_iter = 12000

x_train = np.load('rotrate_x_1.npy')
w_train = np.load('rotrate_u_1.npy')

[x_train,u_train] = dataprocess(x_train,w_train)

u_train[:,:,1:] = x_train[:,:-1,9:]

x_train = x_train[:,:,:9]

np.save('rotrate_mod_x_1.npy', x_train)
np.save('rotrate_mod_u_1.npy', u_train)



