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

modelnum = 4
trajlen = 50
trajnum = 1000

datalen = int((trajnum/10) + trajlen - 1)

error = np.zeros([modelnum,trajlen])

stddev = np.zeros([modelnum,trajlen])

poserror = np.zeros([modelnum,trajlen])

posstddev = np.zeros([modelnum,trajlen])

besttraj_res_x = np.zeros([modelnum,trajlen,n])
besttraj_ref_x = np.zeros([modelnum,trajlen,n])
besttraj_u = np.zeros([modelnum,trajlen-1,m])

traj_res_x = np.zeros([modelnum,trajlen,n])
traj_ref_x = np.zeros([modelnum,trajlen,n])
traj_u = np.zeros([modelnum,trajlen-1,m])

dataversel = ['3', '4', '5', '6']

for n1 in range(modelnum):

    dataver = dataversel[n1]
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

    poserrortot = np.linalg.norm(deltax[:,:,:3],axis=2)
    poserroravg = np.mean(poserrortot,axis=0)
    posstddevavg = np.std(poserrortot,axis=0)

    error[n1,:] = erroravg
    stddev[n1,:] = stddevavg

    poserror[n1,:] = poserroravg
    posstddev[n1,:] = posstddevavg

    minerror = min(errortot[:,-1])
    minerror_col = np.where(errortot[:,-1] == minerror)

    print('Minimum Error Column:')
    print(minerror_col)

    besttraj_res_x[n1,:,:] = xres[minerror_col,:,:]
    besttraj_ref_x[n1,:,:] = xtrain[minerror_col,:,:]
    besttraj_u[n1,:,:] = utrain[minerror_col,:,:]

    traj_res_x[n1,:,:] = xres[0,:,:]
    traj_ref_x[n1,:,:] = xtrain[0,:,:]
    traj_u[n1,:,:] = utrain[0,:,:]

yawsel = ['11.25', '22.5', '45', '90']
yawvec = [11.25, 22.5, 45, 90]

fig1, ax1 = plt.subplots(2,2)
m1 = 0
for n1 in np.arange(2):
    for n2 in np.arange(2):
        ax1[n1,n2].plot(np.arange(trajlen),error[m1,:],'k-')
        ax1[n1,n2].fill_between(np.arange(trajlen), error[m1,:]+stddev[m1,:], error[m1,:]-stddev[m1,:])
        ax1[n1,n2].set(xlabel='Steps')
        ax1[n1,n2].set(ylabel='Error')
        Title = 'Yaw = ' + yawsel[m1] + ' deg'
        ax1[n1,n2].set_title(Title)
        m1 = m1 + 1
plt.show()

fig1, ax1 = plt.subplots(2,2)
m1 = 0
for n1 in np.arange(2):
    for n2 in np.arange(2):
        ax1[n1,n2].plot(np.arange(trajlen),poserror[m1,:],'k-')
        ax1[n1,n2].fill_between(np.arange(trajlen), poserror[m1,:]+posstddev[m1,:], poserror[m1,:]-posstddev[m1,:])
        ax1[n1,n2].set(xlabel='Steps')
        ax1[n1,n2].set(ylabel='Position Error')
        Title = 'Yaw = ' + yawsel[m1] + ' deg'
        ax1[n1,n2].set_title(Title)
        m1 = m1 + 1
plt.show()

plt.figure()
plt.plot(yawvec, error[:,-1],'k-o')
plt.fill_between(yawvec, error[:,-1]+stddev[:,-1], error[:,-1]-stddev[:,-1])
plt.xlabel('Yaw (deg)')
plt.ylabel('Error')
plt.title('Error vs Yaw (50 steps)')
plt.show()

plt.figure()
plt.plot(yawvec, error[:,24],'k-o')
plt.fill_between(yawvec, error[:,24]+stddev[:,24], error[:,24]-stddev[:,24])
plt.xlabel('Yaw (deg)')
plt.ylabel('Error')
plt.title('Error vs Yaw (25 steps)')
plt.show()

plt.figure()
plt.plot(yawvec, poserror[:,-1],'k-o')
plt.fill_between(yawvec, poserror[:,-1]+posstddev[:,-1], poserror[:,-1]-posstddev[:,-1])
plt.xlabel('Yaw (deg)')
plt.ylabel('Position Error')
plt.title('Position Error vs Yaw (50 steps)')
plt.show()

plt.figure()
plt.plot(yawvec, poserror[:,24],'k-o')
plt.fill_between(yawvec, poserror[:,24]+posstddev[:,24], poserror[:,24]-posstddev[:,24])
plt.xlabel('Yaw (deg)')
plt.ylabel('Position Error')
plt.title('Position Error vs Yaw (25 steps)')
plt.show()

figelement = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'dx', 'dy', 'dz', 'T', 'omega_x', 'omega_y', 'omega_z']

fig1, ax1 = plt.subplots(14,4,sharex=True)
m1 = 0
for n1 in np.arange(4):
    for n2 in np.arange(14):
        if n2 <= 9:
            if n2 == 0 and n1 == 0:
                ax1[n2,n1].plot(np.arange(trajlen), besttraj_res_x[n1,:,n2], 'r-', label='Koopman')
                ax1[n2,n1].plot(np.arange(trajlen), besttraj_ref_x[n1,:,n2], 'k-', label='Reference')
            else:
                ax1[n2,n1].plot(np.arange(trajlen), besttraj_res_x[n1,:,n2], 'r-')
                ax1[n2,n1].plot(np.arange(trajlen), besttraj_ref_x[n1,:,n2], 'k-')
        else:
            if n2 == 10 and n1 == 0:
                ax1[n2,n1].stairs(besttraj_u[n1,:,n2-10], np.arange(trajlen), label='Control Input')
            else:
                ax1[n2,n1].stairs(besttraj_u[n1,:,n2-10], np.arange(trajlen))
        if n1 == 0:
            ax1[n2,n1].set(ylabel=figelement[n2])
        if n2 == 0:
            Title = 'yaw = ' + yawsel[n1] + ' deg'
            ax1[n2,n1].set_title(Title)
fig1.legend()
fig1.supxlabel('Steps')
fig1.suptitle('Best Case')
plt.show()

fig1, ax1 = plt.subplots(14,4,sharex=True)
m1 = 0
for n1 in np.arange(4):
    for n2 in np.arange(14):
        if n2 <= 9:
            if n2 == 0 and n1 == 0:
                ax1[n2,n1].plot(np.arange(trajlen), traj_res_x[n1,:,n2], 'r-', label='Koopman')
                ax1[n2,n1].plot(np.arange(trajlen), traj_ref_x[n1,:,n2], 'k-', label='Reference')
            else:
                ax1[n2,n1].plot(np.arange(trajlen), traj_res_x[n1,:,n2], 'r-')
                ax1[n2,n1].plot(np.arange(trajlen), traj_ref_x[n1,:,n2], 'k-')
        else:
            if n2 == 10 and n1 == 0:
                ax1[n2,n1].stairs(traj_u[n1,:,n2-10], np.arange(trajlen), label='Control Input')
            else:
                ax1[n2,n1].stairs(traj_u[n1,:,n2-10], np.arange(trajlen))
        if n1 == 0:
            ax1[n2,n1].set(ylabel=figelement[n2])
        if n2 == 0:
            Title = 'yaw = ' + yawsel[n1] + ' deg'
            ax1[n2,n1].set_title(Title)
fig1.legend()
fig1.supxlabel('Steps')
fig1.suptitle('Typical Case')
plt.show()



