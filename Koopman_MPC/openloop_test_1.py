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

from func_simulation import simulate_koop, simulate_euler

from func_attconv import eultorotvec, rotvectoeul

import os

def reshape_sw(olddata,newlen,dim):
    oldtrajnum = len(olddata[:,0,0])
    oldlen = len(olddata[0,:,0])
    newtrajnum = oldtrajnum*(oldlen-newlen+1)

    newdata = np.zeros((newtrajnum,newlen,dim))

    print('Old Data Shape: ')
    print(np.shape(olddata))
    print('New Data Shape: ')
    print(np.shape(newdata))

    iter = 0
    for n1 in range(oldtrajnum):
        for n2 in range(oldlen-newlen+1):
            for n3 in range(dim):
                newdata[iter,:,n3] = olddata[n1,n2:n2+newlen,n3]
            iter = iter + 1
            
    return newdata

def defkbf(n,m,len):
    return CKBF_STK([n, m, 12+1], 100*np.ones(len), True, jax.nn.swish)

n = 10
m = 4

modelnum = 1
datalen = 100
trajlen = 50

trajnum = 10*(datalen-trajlen+1)

error = np.zeros([modelnum,trajlen])

stddev = np.zeros([modelnum,trajlen])

for n1 in range(modelnum):

    dataver = '1'
    modelver  = '1'

    koopmat_blo = np.load(os.getcwd() + '/trainmodel/crazyflie_' + dataver + '_matrices_quat_blo_' + modelver + '.npy')
    with open(os.getcwd() + '/trainmodel/crazyflie_' + dataver + '_net_quat_blo_' + modelver + '.pkl','rb') as fp:
        params_blo = pickle.load(fp)

    xscal = np.load(os.getcwd() + '/trainmodel/crazyflie_' + dataver + '_xscal_quat_blo_' + modelver + '.npy')
    uscal = np.load(os.getcwd() + '/trainmodel/crazyflie_' + dataver + '_uscal_quat_blo_' + modelver + '.npy')

    kbf = defkbf(n,m,2)
    def basis_blo(x):
        return np.array(kbf.encoder(x,params_blo))
    
    xtrain = np.load(os.getcwd() + '/traindata/crazyflie_x_quat_train_' + dataver + '.npy')
    utrain = np.load(os.getcwd() + '/traindata/crazyflie_u_train_quad_' + dataver + '.npy')

    xtrain = xtrain[:, :datalen, :]
    utrain = utrain[:, :datalen-1, :]

    xtrain = reshape_sw(xtrain, trajlen, n)
    utrain = reshape_sw(utrain, trajlen-1, m)

    xres = np.zeros([trajnum, trajlen, n])

    for n2 in range(trajnum):

        print(n2)

        xres[n2,0,:] = xtrain[n2,0,:]
        utest = utrain[n2,:,:]

        for n3 in range(1,trajlen):
            [xres[n2,n3,:], unext] = simulate_koop(np.squeeze(xres[n2,n3-1,:]),np.squeeze(utest[n3-1,:]),koopmat_blo,basis_blo,xscal,uscal,0.02,n)
        
    deltax = xtrain - xres
    errortot = np.linalg.norm(deltax,axis=2)
    erroravg = np.mean(errortot,axis=0)
    stddevavg = np.std(errortot,axis=0)

    error[n1,:] = erroravg
    stddev[n1,:] = stddevavg

plt.figure()
plt.plot(np.arange(trajlen), error[0,:], 'k-', label='yaw = +/- 180 deg')
plt.fill_between(np.arange(trajlen), error[0,:]+stddev[0,:], error[0,:]-stddev[0,:])
plt.xlabel('Steps')
plt.ylabel('Error')
plt.legend()
plt.show()






