from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, pi
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as Rot

from scipy.integrate import solve_ivp

from KoopmanMPC_CasADi_Class_1 import Quad_KoopmanMPC_Euler

from Cont_func_1 import CKBF_STK

import jax

import pickle

import do_mpc

def quadmpc_koop(xarray,zarray,n,koopmat,modeltype,basis,xscal,uscal):
    t1 = time()

    def funcxnorm(x,scal,dim):
        xnorm = np.zeros(dim)
        for n1 in range(dim):
            xnorm[n1] = (x[n1] - scal[n1,1])/scal[n1,0]
        return xnorm
    
    def funcxscal(xnorm,scal,dim):
        x = np.zeros(dim)
        for n1 in range(dim):
            x[n1] = (xnorm[n1]*scal[n1,0]) + scal[n1,1]
        return x

    datalen = len(xarray[:,0])

    itertime = np.zeros(datalen)

    contw = np.zeros(n)
    contw[0:3] = 20.0
    contw[3:6] = 0.02
    contw[6:9] = 1.0
    contw[9:12] = 0.02

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

    u0_norm = funcxnorm(uhover_norm, uscal, 4)

    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    x = model.set_variable(var_type='_x', var_name='x', shape=(1,1))
    y = model.set_variable(var_type='_x', var_name='y', shape=(1,1))
    z = model.set_variable(var_type='_x', var_name='z', shape=(1,1))
    phi = model.set_variable(var_type='_x', var_name='phi', shape=(1,1))
    theta = model.set_variable(var_type='_x', var_name='theta', shape=(1,1))
    psi = model.set_variable(var_type='_x', var_name='psi', shape=(1,1))
    dx = model.set_variable(var_type='_x', var_name='dx', shape=(1,1))
    dy = model.set_variable(var_type='_x', var_name='dy', shape=(1,1))
    dz = model.set_variable(var_type='_x', var_name='dz', shape=(1,1))
    Th = model.set_variable(var_type='_u', var_name='Th')
    tx = model.set_variable(var_type='_u', var_name='tx')
    ty = model.set_variable(var_type='_u', var_name='ty')
    tz = model.set_variable(var_type='_u', var_name='tz')

    xref1 = model.set_variable('_tvp', 'xref1', shape=(12,1))

    model.set_rhs('x', dx)
    model.set_rhs('y', dy)
    model.set_rhs('z', dz)
    model.set_rhs('phi', tx)
    model.set_rhs('theta', ty)
    model.set_rhs('psi', tz)

    dx_next = ((cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi))/mass)*(Th*0.1*max_thrust)
    dy_next = ((sin(psi)*sin(theta)*cos(phi) - cos(psi)*sin(phi))/mass)*(Th*0.1*max_thrust)
    dz_next = -gravity + ((cos(theta)*cos(phi))/mass)*(Th*0.1*max_thrust)

    model.set_rhs('dx', dx_next)
    model.set_rhs('dy', dy_next)
    model.set_rhs('dz', dz_next)

    # Build the model
    model.setup()

    model_type_koop = modeltype # either 'discrete' or 'continuous'
    model_koop = do_mpc.model.Model(model_type_koop)

    z = model_koop.set_variable(var_type='_x', var_name='z', shape=(n,1))
    Th = model_koop.set_variable(var_type='_u', var_name='Th')
    tx = model_koop.set_variable(var_type='_u', var_name='tx')
    ty = model_koop.set_variable(var_type='_u', var_name='ty')
    tz = model_koop.set_variable(var_type='_u', var_name='tz')

    zref1 = model_koop.set_variable('_tvp', 'zref1', shape=(n,1))

    G1 = koopmat[1,:,:]
    G2 = koopmat[2,:,:]
    G3 = koopmat[3,:,:]
    G4 = koopmat[4,:,:]

    x_next = z + dt*(koopmat[0,:,:] + ((G1)*(Th) + (G2)*(tx) + (G3)*(ty) + (G4)*(tz)))@z

    model_koop.set_rhs('z', x_next)

    # Build the model
    model_koop.setup()

    mpc = do_mpc.controller.MPC(model_koop)

    if modeltype == 'discrete':
        setup_mpc = {
            'n_horizon': 25,
            #'n_horizon': 10,
            't_step': 0.02,
            'n_robust': 0,
            #'n_robust': 2,
            'state_discretization': 'discrete',
            'store_full_solution': True,
            'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
        }
    else:
        setup_mpc = {
        'n_horizon': 25,
        #'n_horizon': 10,
        't_step': 0.02,
        'n_robust': 0,
        #'n_robust': 1,
        'state_discretization': 'collocation',
        'collocation_type': 'radau',
        'collocation_deg': 2,
        'collocation_ni': 1,
        'store_full_solution': True,
        'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
        }

    mpc.set_param(**setup_mpc)

    mpc.settings.supress_ipopt_output()

    errorpen = 1.0

    #pospen = 40.0
    #pospen = 10.0
    pospen = 1.0
    xpen = contw[0]
    ypen = contw[1]
    zpen = contw[2]

    #angpen = 0.1
    #angpen = 0.2
    angpen = 1.0
    phipen = contw[3]
    thetapen = contw[4]
    psipen = contw[5]

    #velpen = 2.0
    velpen = 1.0
    dxpen = contw[6]
    dypen = contw[7]
    dzpen = contw[8]

    #angvelpen = 0.1
    #angvelpen = 0.2
    angvelpen = 1.0
    dphipen = contw[9]
    dthetapen = contw[10]
    dpsipen = contw[11]

    cost = pospen*(xpen*(z[1] - model_koop.tvp['zref1',1])**2 + ypen*(z[2] - model_koop.tvp['zref1',2])**2 + zpen*(z[3] - model_koop.tvp['zref1',3])**2) + angpen*(phipen*(z[4] - model_koop.tvp['zref1',4])**2 + thetapen*(z[5] - model_koop.tvp['zref1',5])**2 + psipen*(z[6] - model_koop.tvp['zref1',6])**2) + velpen*(dxpen*(z[7] - model_koop.tvp['zref1',7])**2 + dypen*(z[8] - model_koop.tvp['zref1',8])**2 + dzpen*(z[9] - model_koop.tvp['zref1',9])**2)
    mpc.set_objective(mterm=0*cost, lterm=cost)

    inputpen = 1.0

    mpc.set_rterm(Th=inputpen,tx=inputpen,ty=inputpen,tz=inputpen) # input penalty

    max_x = np.array([10.0, 10.0, 40.0, np.pi/2, np.pi/2, np.pi/2, 20.0, 20.0, 20.0])
    min_x = np.array([-10.0, -10.0, 0.0, -np.pi/2, -np.pi/2, -np.pi/2, -20.0, -20.0, -20.0])
    max_z = (np.inf)*(np.ones(n))
    min_z = (-np.inf)*(np.ones(n))

    max_z[1:10] = funcxnorm(max_x, xscal, 9)
    min_z[1:10] = funcxnorm(min_x, xscal, 9)

    # lower bounds of the states
    mpc.bounds['lower','_x','z'] = min_z

    # upper bounds of the states
    mpc.bounds['upper','_x','z'] = max_z

    umax = np.array([10.0, 2.0, 2.0, 2.0])
    umin = np.array([0.0, -2.0, -2.0, -2.0])

    umaxnorm = funcxnorm(umax, uscal, 4)
    uminnorm = funcxnorm(umin, uscal, 4)

    # lower bounds of the input
    mpc.bounds['lower','_u','Th'] = uminnorm[0]
    mpc.bounds['lower','_u','tx'] = uminnorm[1]
    mpc.bounds['lower','_u','ty'] = uminnorm[2]
    mpc.bounds['lower','_u','tz'] = uminnorm[3]

    # upper bounds of the input
    mpc.bounds['upper','_u','Th'] = umaxnorm[0]
    mpc.bounds['upper','_u','tx'] = umaxnorm[1]
    mpc.bounds['upper','_u','ty'] = umaxnorm[2]
    mpc.bounds['upper','_u','tz'] = umaxnorm[3]

    tvp_template = mpc.get_tvp_template()

    def tvp_fun(t_now):
        n2 = int(t_now/0.02) + 1
        #n2 = int(t_now/0.02)
        for n3 in range(n):
            tvp_template['_tvp',:, 'zref1', n3] = zarray[n2,n3]
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)

    mpc.setup()

    simulator = do_mpc.simulator.Simulator(model)

    simulator.set_param(t_step = 0.02)

    tvp_template_sim = simulator.get_tvp_template()

    def tvp_fun_sim(t_now):
        return tvp_template_sim

    simulator.set_tvp_fun(tvp_fun_sim)

    simulator.setup()

    u0 = uhover_norm

    # Initial state
    x0 = xarray[0,:]
    z0 = zarray[0,:]
    mpc.x0 = z0
    simulator.x0 = x0
    mpc.u0 = u0_norm
    simulator.u0 = uhover_norm

    # Use initial state to set the initial guess.
    mpc.set_initial_guess()

    iter = 0
    for n in range(datalen):
        t2 = time()
        iter = iter + 1
        print(iter)
        u0_norm = mpc.make_step(z0)
        u0 = funcxscal(u0,uscal,4)
        x0 = simulator.make_step(np.reshape(u0,(4,1)))
        print(np.linalg.norm(x0[:3] - xarray[n+1,:3]))
        x0 = np.squeeze(x0)
        z0 = np.transpose(basis(funcxnorm(x0,xscal,9)))
        itertime[iter-1] = time() - t2
        if x0[2] < 0.0:
            print('z below 0.0')
            break

    results1 = mpc.data
    t_f = results1['_time']
    z_f = results1['_x']
    #u_f = results1['_u']

    t_f = np.squeeze(t_f,axis=1)

    results2 = simulator.data
    x_f = results2['_x']
    u_f = results2['_u']

    u_f = u_f[:-1,:]

    return [t_f,x_f,u_f,itertime]


def simulate_euler(step_horizon, t0, state_init, u):

    u_ctrl = np.squeeze(u[:,1])

    def lin_func(t, x):
        xdot = [x[6], x[7], x[8], u_ctrl[1] ,u_ctrl[2], u_ctrl[3],
                ((cos(x[5])*sin(x[4])*cos(x[3]) + sin(x[5])*sin(x[3]))/mass)*(u_ctrl[0]*0.1*max_thrust),
                ((sin(x[5])*sin(x[4])*cos(x[3]) - cos(x[5])*sin(x[3]))/mass)*(u_ctrl[0]*0.1*max_thrust),
                -gravity + ((cos(x[4])*cos(x[3]))/mass)*(u_ctrl[0]*0.1*max_thrust)]
        return xdot
    soln = solve_ivp(lin_func, [0.0, step_horizon], np.squeeze(state_init), t_eval = [step_horizon])
    xnext = soln.y
    next_state_euler =  np.squeeze(xnext)

    next_state_quat = np.zeros(10)
    next_state_quat[:3] = next_state_euler[:3]
    next_state_quat[3:7] = Rot.from_euler('xyz', next_state_euler[3:6]).as_quat()
    next_state_quat[7:] = next_state_euler[6:]

    return next_state_euler, u_ctrl

def defkbf(n,m):
    return CKBF_STK([n, m, 12+1], 100*np.ones(2), True, jax.nn.swish)

n = 9
m = 4

quadspec = [0.0336, 1.7e-5, 1.7e-5, 2.9e-5, 0.04, 3.1582e-10, 7.9379e-12, 0.057*9.81]

mass = quadspec[0]
Ixx = quadspec[1]
Iyy = quadspec[2]
Izz = quadspec[3]
gravity = 9.81
l = quadspec[4]
ct = quadspec[5]
cq = quadspec[6]

c1 = (Iyy - Izz)/Ixx
c2 = (Izz - Ixx)/Iyy
c3 = (Iyy - Ixx)/Izz

hover_thrust = mass*gravity
max_thrust = quadspec[7]
min_thrust = 0.0

hover_w2 = (hover_thrust)/(4*ct)
max_w2 = (max_thrust)/(4*ct)
max_tx = ct*l*max_w2
max_ty = ct*l*max_w2
max_tz = 2*cq*max_w2

dt = 0.02
#dt = 0.05

model = 'BLO'

if model == 'BLO':
    koopmat_blo = np.load('matrices_rotrate_crazyflie_blo_1.npy')
    with open('net_rotrate_crazyflie_blo_1.pkl','rb') as fp:
        params_blo = pickle.load(fp)

    xscal = np.load('xscal_rotrate_crazyflie_blo_1.npy')
    uscal = np.load('uscal_rotrate_crazyflie_blo_1.npy')

    kbf = defkbf(n,m)
    def basis_blo(x):
        return np.array(kbf.encoder(x,params_blo))
    
'''
xtrain = np.load('x_quat_train_3.npy')

for n1 in range(n):
    maxval = np.max(np.abs(xtrain[:,:,n1]))
    if n1 == 2:
        xscal[n1] = 0.1*maxval
    else:
        xscal[n1] = 0.2*maxval
'''

# Define Trajectory

ti = 0.0
tf = 4.0

tracking = 1
mindist = 0

trajlen = int((tf-ti)/dt)

xinit = np.array([0.0, 0.0, 0.5,
                  0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0,])

xtarg = np.array([0.0, 0.0, 0.5,
                  0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0,])

dxdt = (xtarg[0] - xinit[0])/(tf-ti)
dydt = (xtarg[1] - xinit[1])/(tf-ti)
dzdt = (xtarg[2] - xinit[2])/(tf-ti)

if tracking == 1:
    xtarg[6] = dxdt
    xtarg[7] = dydt
    xtarg[8] = dzdt

xarray_targ = np.zeros((trajlen+1,n))
xarray_targ[0,:] = xinit

for n1 in range(trajlen):
    xarray_targ[n1+1,:] = xtarg
    if tracking == 1:
        xarray_targ[n1+1,0] = xarray_targ[0,0] + dxdt*(n1+1)*dt
        xarray_targ[n1+1,1] = xarray_targ[0,1] + dydt*(n1+1)*dt
        xarray_targ[n1+1,2] = xarray_targ[0,2] + dzdt*(n1+1)*dt

zarray_targ = np.zeros((trajlen+1,23))

for n1 in range(trajlen+1):
    zarray_targ[n1,:] = basis_blo((xarray_targ[n1,:]-xscal[:,1])/xscal[:,0])

[tarray,xarray_euler,uarray,itertime_blo] = quadmpc_koop(xarray_targ,zarray_targ,23,koopmat_blo,'discrete',basis_blo,xscal,uscal)

plt.figure()
plt.plot(tarray, xarray_euler[:,0])
plt.xlabel('Time (s)')
plt.ylabel('x (m)')
plt.show()

plt.figure()
plt.plot(tarray, xarray_euler[:,1])
plt.xlabel('Time (s)')
plt.ylabel('y (m)')
plt.show()

plt.figure()
plt.plot(tarray, xarray_euler[:,2])
plt.xlabel('Time (s)')
plt.ylabel('z (m)')
plt.show()

plt.figure()
plt.plot(tarray, (xarray_euler[:,3]/np.pi)*180, label='phi')
plt.plot(tarray, (xarray_euler[:,4]/np.pi)*180, label='theta')
plt.plot(tarray, (xarray_euler[:,5]/np.pi)*180, label='psi')
plt.xlabel('Time (s)')
plt.ylabel('Angle (deg)')
plt.legend()
plt.show()

'''
plt.figure()
plt.plot(tarray, xarray[:,3], label='qx')
plt.plot(tarray, xarray[:,4], label='qy')
plt.plot(tarray, xarray[:,5], label='qz')
plt.plot(tarray, xarray[:,6], label='qw')
plt.xlabel('Time (s)')
plt.ylabel('Quaternion')
plt.legend()
plt.show()
'''

plt.figure()
plt.stairs(uarray[:,0]*0.1*max_thrust, tarray, label='Thrust')
plt.plot([tarray[0],tarray[-1]], hover_thrust*np.ones(2), '--', label='Hover Thrust')
plt.plot([tarray[0],tarray[-1]], max_thrust*np.ones(2), '--', label='Max Thrust')
plt.xlabel('Time (s)')
plt.ylabel('Thrust (N)')
plt.legend()
plt.show()

plt.figure()
plt.stairs(uarray[:,1], tarray, label='omega_x')
plt.stairs(uarray[:,2], tarray, label='omega_y')
plt.stairs(uarray[:,3], tarray, label='omega_z')
plt.xlabel('Time (s)')
plt.ylabel('Rotation Rate (rad/s)')
plt.legend()
plt.show()

plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(xarray_euler[:,0], xarray_euler[:,1], xarray_euler[:,2], 'c-', label='Trajectory')
ax.plot3D(xtarg[0], xtarg[1], xtarg[2], 'o', label='Target')
plt.legend()
ax.set_xlabel('x, m')
ax.set_ylabel('y, m')
ax.set_zlabel('z, m')
plt.show()



