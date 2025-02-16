import argparse
import jax.numpy as jn
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
import scipy.integrate as spi

from Disc_func_1 import fit_model, plt_hist

from Cont_func_1 import CKBF_STK, make_slo_loss

from Cont_func_2 import make_disc_slo_loss

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

def BLOfunc(t_train,x_train,u_train,net_params,disc = 0,len_sw = 0):
    
    BATCH_SIZE = net_params['batch_size']
    HORIZON_LENGTH = len(t_train)
    if disc == 1:
        INT_ORDER = 0
    else:
        #INT_ORDER = 4
        INT_ORDER = 1
    NUM_TRAIN_STEPS = net_params['epochs']
    suf = f'ci_b{BATCH_SIZE}h{HORIZON_LENGTH}'

    n = net_params['state_dim']
    m = net_params['ctrl_dim']
    nz = net_params['encoder_output_dim'] + net_params['state_dim'] + 1

    cas = 'slo'

    kbf = CKBF_STK([n, m, net_params['encoder_output_dim']+1], net_params['encoder_hidden_width']*np.ones(net_params['encoder_hidden_depth']), True, jax.nn.swish)

    make_loss = make_slo_loss

    if len_sw >= 2:
        HORIZON_LENGTH = len_sw
        t_train = t_train[:len_sw]
        x_train = reshape_sw(x_train,len_sw,n)
        u_train = reshape_sw(u_train,len_sw,m)

    TRAINING_DATA = np.zeros((len(x_train[:,0,0]),HORIZON_LENGTH,n+m))
    TRAINING_DATA[:,:,0:n] = x_train
    TRAINING_DATA[:,:,n:n+m] = u_train

    TRAINING_DATA = np.reshape(TRAINING_DATA,(len(x_train[:,0,0])//BATCH_SIZE,BATCH_SIZE,HORIZON_LENGTH,n+m))

    print(np.shape(TRAINING_DATA))

    lss, rst = make_loss(kbf, HORIZON_LENGTH, INT_ORDER, net_params['dt'])

    optimizer = optax.adam(learning_rate=net_params['lr'])

    fit = fit_model(f'res/{cas}_{suf}', lss, TRAINING_DATA, NUM_TRAIN_STEPS, 200000, len(TRAINING_DATA[:,0,0,0])-1, reset=rst, init_only=False)
    ini = kbf.init_params()
    params, hist = fit(ini, optimizer)

    f = plt_hist(hist, ['L', 'Lr', 'Ld'], ['b-', 'r--', 'k--'])
    plt.show()

    Atotal = params['As']

    A_blo = np.zeros((nz,nz))
    A_blo[:,:] = Atotal[:,:nz]

    B_blo = np.zeros((m,nz,nz))

    y1 = 0
    for n1 in range(m):
        y1 = y1 + 1
        B_blo[n1,:,:] = Atotal[:,(nz*y1):(nz*(y1+1))]

    if disc == 1:
        A_blo_disc = np.zeros(np.shape(A_blo))
        B_blo_disc = np.zeros(np.shape(B_blo))
        A_blo_disc = np.identity(nz) + A_blo*(net_params['dt'])
        for n1 in range(m):
            B_blo_disc[n1,:,:] = B_blo[n1,:,:]*(net_params['dt'])
        A_blo = A_blo_disc
        B_blo = B_blo_disc

    def basis_blo(x):
        return np.array(kbf.encoder(x,params))

    return A_blo,B_blo,basis_blo

def BLOfunc_save(t_train,x_train,u_train,net_params,twostep = 0, len_sw = 1):
    
    BATCH_SIZE = net_params['batch_size']
    HORIZON_LENGTH = len(t_train)
    INT_ORDER = 1
    NUM_TRAIN_STEPS = net_params['epochs']
    suf = f'ci_b{BATCH_SIZE}h{HORIZON_LENGTH}'

    n = net_params['state_dim']
    m = net_params['ctrl_dim']
    nz = net_params['encoder_output_dim'] + net_params['state_dim'] + 1

    cas = 'slo'

    kbf = CKBF_STK([n, m, net_params['encoder_output_dim']+1], net_params['encoder_hidden_width']*np.ones(net_params['encoder_hidden_depth']), True, jax.nn.swish)

    make_loss = make_slo_loss

    if twostep==1:
        len_sw = 2

    if len_sw >= 2:
        HORIZON_LENGTH = len_sw
        t_train = t_train[:len_sw]
        x_train = reshape_sw(x_train,len_sw,n)
        u_train = reshape_sw(u_train,len_sw-1,m)

    TRAINING_DATA = np.zeros((len(x_train[:,0,0]),HORIZON_LENGTH,n+m))
    TRAINING_DATA[:,:,0:n] = x_train
    TRAINING_DATA[:,:,n:n+m] = u_train

    TRAINING_DATA = np.reshape(TRAINING_DATA,(len(x_train[:,0,0])//BATCH_SIZE,BATCH_SIZE,HORIZON_LENGTH,n+m))

    print(np.shape(TRAINING_DATA))

    lss, rst = make_loss(kbf, HORIZON_LENGTH, INT_ORDER, net_params['dt'])

    optimizer = optax.adam(learning_rate=net_params['lr'])

    fit = fit_model(f'res/{cas}_{suf}', lss, TRAINING_DATA, NUM_TRAIN_STEPS, 200000, len(TRAINING_DATA[:,0,0,0])-1, reset=rst, init_only=False)
    ini = kbf.init_params()
    params, hist = fit(ini, optimizer)

    f = plt_hist(hist, ['L', 'Lr', 'Ld'], ['b-', 'r--', 'k--'])
    plt.show()

    Atotal = params['As']

    A_blo = np.zeros((nz,nz))
    A_blo[:,:] = Atotal[:,:nz]

    B_blo = np.zeros((m,nz,nz))

    y1 = 0
    for n1 in range(m):
        y1 = y1 + 1
        B_blo[n1,:,:] = Atotal[:,(nz*y1):(nz*(y1+1))]

    return A_blo,B_blo,params,hist

def BLOdiscfunc(t_train,x_train,u_train,net_params):
    
    BATCH_SIZE = net_params['batch_size']
    HORIZON_LENGTH = len(t_train)
    INT_ORDER = 1
    NUM_TRAIN_STEPS = net_params['epochs']
    suf = f'ci_b{BATCH_SIZE}h{HORIZON_LENGTH}'

    twostep = 1

    n = net_params['state_dim']
    m = net_params['ctrl_dim']
    nz = net_params['encoder_output_dim'] + net_params['state_dim'] + 1

    cas = 'slo'

    kbf = CKBF_STK([n, m, net_params['encoder_output_dim']+1], net_params['encoder_hidden_width']*np.ones(net_params['encoder_hidden_depth']), True, jax.nn.swish)

    make_loss = make_disc_slo_loss

    if twostep == 1:
        new_x_train = np.zeros((len(x_train[:,0,0])*(HORIZON_LENGTH-1),2,n))
        new_u_train = np.zeros((len(u_train[:,0,0])*(HORIZON_LENGTH-1),2,m))
        iter = 0
        for n1 in range(len(x_train[:,0,0])):
            for n2 in range(HORIZON_LENGTH-1):
                new_x_train[iter,0,:] = x_train[n1,n2,:]
                new_x_train[iter,1,:] = x_train[n1,n2+1,:]
                for n3 in range(m):
                    new_u_train[iter,:,n3] = u_train[n1,n2,n3]
                iter = iter + 1
        HORIZON_LENGTH = 2
        t_train = t_train[:2]
        x_train = new_x_train
        u_train = new_u_train

    TRAINING_DATA = np.zeros((len(x_train[:,0,0]),HORIZON_LENGTH,n+m))
    TRAINING_DATA[:,:,0:n] = x_train
    TRAINING_DATA[:,:,n:n+m] = u_train

    TRAINING_DATA = np.reshape(TRAINING_DATA,(len(x_train[:,0,0])//BATCH_SIZE,BATCH_SIZE,HORIZON_LENGTH,n+m))

    print(np.shape(TRAINING_DATA))

    lss, rst = make_loss(kbf, HORIZON_LENGTH, INT_ORDER, net_params['dt'])

    optimizer = optax.adam(learning_rate=net_params['lr'])

    fit = fit_model(f'res/{cas}_{suf}', lss, TRAINING_DATA, NUM_TRAIN_STEPS, 200000, len(TRAINING_DATA[:,0,0,0])-1, reset=rst, init_only=False)
    ini = kbf.init_params()
    params, hist = fit(ini, optimizer)

    f = plt_hist(hist, ['L', 'Lr', 'Ld'], ['b-', 'r--', 'k--'])
    plt.show()

    Atotal = params['As']

    A_blo = np.zeros((nz,nz))
    A_blo[:,:] = Atotal[:,:nz]

    B_blo = np.zeros((m,nz,nz))

    y1 = 0
    for n1 in range(m):
        y1 = y1 + 1
        B_blo[n1,:,:] = Atotal[:,(nz*y1):(nz*(y1+1))]

    def basis_blo(x):
        return np.array(kbf.encoder(x,params))

    return A_blo,B_blo,basis_blo

def BLOdiscfunc_save(t_train,x_train,u_train,net_params):
    
    BATCH_SIZE = net_params['batch_size']
    HORIZON_LENGTH = len(t_train)
    INT_ORDER = 1
    NUM_TRAIN_STEPS = net_params['epochs']
    suf = f'ci_b{BATCH_SIZE}h{HORIZON_LENGTH}'

    twostep = 1

    n = net_params['state_dim']
    m = net_params['ctrl_dim']
    nz = net_params['encoder_output_dim'] + net_params['state_dim'] + 1

    cas = 'slo'

    kbf = CKBF_STK([n, m, net_params['encoder_output_dim']+1], net_params['encoder_hidden_width']*np.ones(net_params['encoder_hidden_depth']), True, jax.nn.swish)

    make_loss = make_disc_slo_loss

    if twostep == 1:
        new_x_train = np.zeros((len(x_train[:,0,0])*(HORIZON_LENGTH-1),2,n))
        new_u_train = np.zeros((len(u_train[:,0,0])*(HORIZON_LENGTH-1),2,m))
        iter = 0
        for n1 in range(len(x_train[:,0,0])):
            for n2 in range(HORIZON_LENGTH-1):
                new_x_train[iter,0,:] = x_train[n1,n2,:]
                new_x_train[iter,1,:] = x_train[n1,n2+1,:]
                for n3 in range(m):
                    new_u_train[iter,:,n3] = u_train[n1,n2,n3]
                iter = iter + 1
        HORIZON_LENGTH = 2
        t_train = t_train[:2]
        x_train = new_x_train
        u_train = new_u_train

    TRAINING_DATA = np.zeros((len(x_train[:,0,0]),HORIZON_LENGTH,n+m))
    TRAINING_DATA[:,:,0:n] = x_train
    TRAINING_DATA[:,:,n:n+m] = u_train

    TRAINING_DATA = np.reshape(TRAINING_DATA,(len(x_train[:,0,0])//BATCH_SIZE,BATCH_SIZE,HORIZON_LENGTH,n+m))

    print(np.shape(TRAINING_DATA))

    lss, rst = make_loss(kbf, HORIZON_LENGTH, INT_ORDER, net_params['dt'])

    optimizer = optax.adam(learning_rate=net_params['lr'])

    fit = fit_model(f'res/{cas}_{suf}', lss, TRAINING_DATA, NUM_TRAIN_STEPS, 200000, len(TRAINING_DATA[:,0,0,0])-1, reset=rst, init_only=False)
    ini = kbf.init_params()
    params, hist = fit(ini, optimizer)

    f = plt_hist(hist, ['L', 'Lr', 'Ld'], ['b-', 'r--', 'k--'])
    plt.show()

    Atotal = params['As']

    A_blo = np.zeros((nz,nz))
    A_blo[:,:] = Atotal[:,:nz]

    B_blo = np.zeros((m,nz,nz))

    y1 = 0
    for n1 in range(m):
        y1 = y1 + 1
        B_blo[n1,:,:] = Atotal[:,(nz*y1):(nz*(y1+1))]

    return A_blo,B_blo,params,hist



