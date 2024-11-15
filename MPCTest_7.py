import numpy as np
import sys
from casadi import *

import do_mpc

import matplotlib.pyplot as plt

from matplotlib import rcParams

import time

from scipy.spatial.transform import Rotation as R

def R_ib(q):
    Rmatrix = np.zeros((3,3))
    q = np.squeeze(q)
    s = (1/(np.linalg.norm(q)**2))

    Rmatrix[0,0] = 1 - 2*s*(q[2]**2 + q[3]**2)
    Rmatrix[0,1] = 2*s*(q[1]*q[2] - q[0]*q[3])
    Rmatrix[0,2] = 2*s*(q[1]*q[3] + q[0]*q[2])

    Rmatrix[1,0] = 2*s*(q[1]*q[2] + q[0]*q[3])
    Rmatrix[1,1] = 1 - 2*s*(q[1]**2 + q[3]**2)
    Rmatrix[1,2] = 2*s*(q[2]*q[3] - q[0]*q[1])

    Rmatrix[2,0] = 2*s*(q[1]*q[3] - q[0]*q[2])
    Rmatrix[2,1] = 2*s*(q[2]*q[3] + q[0]*q[1])
    Rmatrix[2,2] = 1 - 2*s*(q[1]**2 + q[2]**2)

    return Rmatrix

def R_bi(q):
    Rmatrix = R_ib(q)
    Rmatrix = np.transpose(Rmatrix)

    return Rmatrix

modeltype = 'quat'
#modeltype = 'euler'

if modeltype == 'quat':
    n = 13
else:
    n = 12

m = 4

'''
mass = 0.0336
Ixx = 1.7e-5
Iyy = 1.7e-5
Izz = 2.9e-5
gravity = 9.81
l = 0.04
ct = 3.1582e-10
cq = 7.9379e-12

hover_w2 = (mass*gravity)/(4*ct)
max_w2 = (0.057*gravity)/(4*ct)
hover_thrust = (mass*gravity)
max_thrust = (0.057*gravity)
max_tx = ct*l*max_w2
max_ty = ct*l*max_w2
max_tz = 2*cq*max_w2
uhover = np.array([[hover_thrust], [0.0], [0.0], [0.0]])
uhover_norm = uhover/(0.1*max_thrust)

c1 = (Iyy - Izz)/Ixx
c2 = (Izz - Ixx)/Iyy
c3 = (Iyy - Ixx)/Izz
'''

mass = 2.02
Ixx = 0.011
Iyy = 0.015
Izz = 0.021
gravity = 9.81
l = 0.26
ct = 5.84e-6
cq = 0.06*ct

c1 = (Iyy - Izz)/Ixx
c2 = (Izz - Ixx)/Iyy
c3 = (Iyy - Ixx)/Izz

hover_thrust = mass*gravity
max_thrust = 28.3
min_thrust = 0.0

hover_w2 = (hover_thrust)/(4*ct)
max_w2 = (max_thrust)/(4*ct)
max_tx = ct*l*max_w2
max_ty = ct*l*max_w2
max_tz = 2*cq*max_w2

dt = 0.02
#dt = 0.04

xtarg = np.zeros(n)

xtarg[0] = 5.0
xtarg[1] = 5.0
xtarg[2] = 5.0

rtarg_euler = R.from_euler('xyz', [0.0, 0.0, 0.0])
rtarg_quat = rtarg_euler.as_quat(scalar_first = True)

xtarg[3:7] = np.array(rtarg_quat)

model_type_quat = 'continuous' # either 'discrete' or 'continuous'
model_quat = do_mpc.model.Model(model_type_quat)

x_quat = model_quat.set_variable(var_type='_x', var_name='x_quat', shape=(1,1))
y_quat = model_quat.set_variable(var_type='_x', var_name='y_quat', shape=(1,1))
z_quat = model_quat.set_variable(var_type='_x', var_name='z_quat', shape=(1,1))
q_w = model_quat.set_variable(var_type='_x', var_name='q_w', shape=(1,1))
q_x = model_quat.set_variable(var_type='_x', var_name='q_x', shape=(1,1))
q_y = model_quat.set_variable(var_type='_x', var_name='q_y', shape=(1,1))
q_z = model_quat.set_variable(var_type='_x', var_name='q_z', shape=(1,1))

dx_quat = model_quat.set_variable(var_type='_x', var_name='dx_quat', shape=(1,1))
dy_quat = model_quat.set_variable(var_type='_x', var_name='dy_quat', shape=(1,1))
dz_quat = model_quat.set_variable(var_type='_x', var_name='dz_quat', shape=(1,1))
omega_x = model_quat.set_variable(var_type='_x', var_name='omega_x', shape=(1,1))
omega_y = model_quat.set_variable(var_type='_x', var_name='omega_y', shape=(1,1))
omega_z = model_quat.set_variable(var_type='_x', var_name='omega_z', shape=(1,1))

Th = model_quat.set_variable(var_type='_u', var_name='Th')
tx = model_quat.set_variable(var_type='_u', var_name='tx')
ty = model_quat.set_variable(var_type='_u', var_name='ty')
tz = model_quat.set_variable(var_type='_u', var_name='tz')

model_quat.set_rhs('x_quat', dx_quat)
model_quat.set_rhs('y_quat', dy_quat)
model_quat.set_rhs('z_quat', dz_quat)

q_w_next = 0.5*(-omega_x*q_x - omega_y*q_y - omega_z*q_z)
q_x_next = 0.5*(omega_x*q_w + omega_y*q_z - omega_z*q_y)
q_y_next = 0.5*(-omega_x*q_z + omega_y*q_w + omega_z*q_x)
q_z_next = 0.5*(omega_x*q_y - omega_y*q_x + omega_z*q_w)

model_quat.set_rhs('q_w', q_w_next)
model_quat.set_rhs('q_x', q_x_next)
model_quat.set_rhs('q_y', q_y_next)
model_quat.set_rhs('q_z', q_z_next)

dx_next_quat = 2*(q_x*q_z - q_w*q_y)*((Th*0.1*max_thrust)/mass)
dy_next_quat = 2*(q_y*q_z + q_w*q_x)*((Th*0.1*max_thrust)/mass)
dz_next_quat = (1 - 2*(q_x**2 + q_y**2))*((Th*0.1*max_thrust)/mass) - gravity

model_quat.set_rhs('dx_quat', dx_next_quat)
model_quat.set_rhs('dy_quat', dy_next_quat)
model_quat.set_rhs('dz_quat', dz_next_quat)

omega_x_next = omega_y*omega_z*c1 + ((tx*0.2*max_tx)/Ixx)
omega_y_next = omega_x*omega_z*c2 + ((ty*0.2*max_tx)/Iyy)
omega_z_next = omega_x*omega_y*c3 + ((tz*0.2*max_tx)/Izz)

model_quat.set_rhs('omega_x', omega_x_next)
model_quat.set_rhs('omega_y', omega_y_next)
model_quat.set_rhs('omega_z', omega_z_next)

# Build the model
model_quat.setup()

model_type_euler = 'continuous' # either 'discrete' or 'continuous'
model_euler = do_mpc.model.Model(model_type_euler)

x_euler = model_euler.set_variable(var_type='_x', var_name='x_euler', shape=(1,1))
y_euler = model_euler.set_variable(var_type='_x', var_name='y_euler', shape=(1,1))
z_euler = model_euler.set_variable(var_type='_x', var_name='z_euler', shape=(1,1))

phi = model_euler.set_variable(var_type='_x', var_name='phi', shape=(1,1))
theta = model_euler.set_variable(var_type='_x', var_name='theta', shape=(1,1))
psi = model_euler.set_variable(var_type='_x', var_name='psi', shape=(1,1))

dx_euler = model_euler.set_variable(var_type='_x', var_name='dx_euler', shape=(1,1))
dy_euler = model_euler.set_variable(var_type='_x', var_name='dy_euler', shape=(1,1))
dz_euler = model_euler.set_variable(var_type='_x', var_name='dz_euler', shape=(1,1))

dphi = model_euler.set_variable(var_type='_x', var_name='dphi', shape=(1,1))
dtheta = model_euler.set_variable(var_type='_x', var_name='dtheta', shape=(1,1))
dpsi = model_euler.set_variable(var_type='_x', var_name='dpsi', shape=(1,1))

Th = model_euler.set_variable(var_type='_u', var_name='Th')
tx = model_euler.set_variable(var_type='_u', var_name='tx')
ty = model_euler.set_variable(var_type='_u', var_name='ty')
tz = model_euler.set_variable(var_type='_u', var_name='tz')

model_euler.set_rhs('x_euler', dx_euler)
model_euler.set_rhs('y_euler', dy_euler)
model_euler.set_rhs('z_euler', dz_euler)

model_euler.set_rhs('phi', dphi)
model_euler.set_rhs('theta', dtheta)
model_euler.set_rhs('psi', dpsi)

dx_next_euler = ((cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi))/mass)*(Th*0.1*max_thrust)
dy_next_euler = ((sin(psi)*sin(theta)*cos(phi) - cos(psi)*sin(phi))/mass)*(Th*0.1*max_thrust)
dz_next_euler = -gravity + ((cos(theta)*cos(phi))/mass)*(Th*0.1*max_thrust)
dphi_next = dtheta*dpsi*c1 + (tx*0.2*max_tx)/Ixx
dtheta_next = dphi*dpsi*c2 + (ty*0.2*max_ty)/Iyy
dpsi_next = dphi*dtheta*c3 + (tz*0.2*max_tz)/Izz

model_euler.set_rhs('dx_euler', dx_next_euler)
model_euler.set_rhs('dy_euler', dy_next_euler)
model_euler.set_rhs('dz_euler', dz_next_euler)
model_euler.set_rhs('dphi', dphi_next)
model_euler.set_rhs('dtheta', dtheta_next)
model_euler.set_rhs('dpsi', dpsi_next)

# Build the model
model_euler.setup()

if modeltype == 'quat':
    mpc = do_mpc.controller.MPC(model_quat)
else:
    mpc = do_mpc.controller.MPC(model_euler)

setup_mpc = {
    'n_horizon': 25,
    't_step': dt,
    'n_robust': 0,
    'store_full_solution': True,
    'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
}

mpc.set_param(**setup_mpc)

mpc.settings.supress_ipopt_output()

costpen = np.array([10.0, 10.0, 10.0, 0.0, 0.0, 0.0])

if modeltype == 'quat':
    cost = costpen[0]*(x_quat - xtarg[0])**2 + costpen[1]*(y_quat - xtarg[1])**2 + costpen[2]*(z_quat - xtarg[2])**2
else:
    cost = costpen[0]*(x_euler - xtarg[0])**2 + costpen[1]*(y_euler - xtarg[1])**2 + costpen[2]*(z_euler - xtarg[2])**2

mpc.set_objective(mterm=0*cost, lterm=cost)

mpc.set_rterm(Th=1.0, tx=1.0, ty=1.0, tz=1.0) # penalty on input changes

# lower bounds of the states
if modeltype == 'quat':
    mpc.bounds['lower', '_x', 'x_quat'] = -20.0
    mpc.bounds['lower', '_x', 'y_quat'] = -20.0
    mpc.bounds['lower', '_x', 'z_quat'] = 0.0
else:
    mpc.bounds['lower', '_x', 'x_euler'] = -20.0
    mpc.bounds['lower', '_x', 'y_euler'] = -20.0
    mpc.bounds['lower', '_x', 'z_euler'] = 0.0

# upper bounds of the states
if modeltype == 'quat':
    mpc.bounds['upper', '_x', 'x_quat'] = 20.0
    mpc.bounds['upper', '_x', 'y_quat'] = 20.0
    mpc.bounds['upper', '_x', 'z_quat'] = 20.0
else:
    mpc.bounds['upper', '_x', 'x_euler'] = 20.0
    mpc.bounds['upper', '_x', 'y_euler'] = 20.0
    mpc.bounds['upper', '_x', 'z_euler'] = 20.0

# lower bounds of the input
mpc.bounds['lower','_u','Th'] = 0.0
mpc.bounds['lower','_u','tx'] = -5.0
mpc.bounds['lower','_u','ty'] = -5.0
mpc.bounds['lower','_u','tz'] = -5.0

# upper bounds of the input
mpc.bounds['upper','_u','Th'] = 10.0
mpc.bounds['upper','_u','tx'] = 5.0
mpc.bounds['upper','_u','ty'] = 5.0
mpc.bounds['upper','_u','tz'] = 5.0

mpc.setup()

if modeltype == 'quat':
    simulator = do_mpc.simulator.Simulator(model_quat)
else:
    simulator = do_mpc.simulator.Simulator(model_euler)

params_simulator = {
    'integration_tool': 'cvodes',
    'abstol': 1e-10,
    'reltol': 1e-10,
    't_step': dt
}

simulator.set_param(**params_simulator)

simulator.setup()

# Initial state

xinit = np.zeros(n)

xinit[0] = 0.0
xinit[1] = 0.0
xinit[2] = 1.0

rinit_euler = R.from_euler('xyz', [0.0, 0.0, 0.0])
rinit_quat = rinit_euler.as_quat(scalar_first = True)

xinit[3:7] = np.array(rinit_quat)

x0 = xinit
u0 = np.array([hover_thrust, 0.0, 0.0, 0.0])

# Set for controller, simulator and estimator
mpc.x0 = x0
mpc.u0 = u0
simulator.x0 = x0
simulator.u0 = u0
mpc.set_initial_guess()

n_steps = 200

tcomp_result = np.zeros(n_steps)

for k in range(n_steps):
    print(k)
    t1 = time.time()
    u0 = mpc.make_step(x0)
    tcomp_result[k] = time.time() - t1
    x0 = simulator.make_step(u0)
    dist = np.linalg.norm(x0[:3] - xtarg[:3])
    if x0[2] < 0.0:
        break
    if dist < 0.1:
        break

results = simulator.data
t_result = results['_time']
x_result = results['_x']
u_result = results['_u']

u_result[:,0] = u_result[:,0]*(0.1*max_thrust)
u_result[:,1] = u_result[:,1]*(0.2*max_tx)
u_result[:,2] = u_result[:,2]*(0.2*max_ty)
u_result[:,3] = u_result[:,3]*(0.2*max_tz)

print('Average Computation Time: ', np.mean(tcomp_result))
print('Max Computation Time: ', np.max(tcomp_result))
print('Min Computation Time: ', np.min(tcomp_result))
print('Computation Time Standard Deviation: ', np.std(tcomp_result))

plt.figure()
plt.plot(t_result,x_result[:,0])
plt.xlabel('Time (s)')
plt.ylabel('x (m)')
plt.show()

plt.figure()
plt.plot(t_result,x_result[:,1])
plt.xlabel('Time (s)')
plt.ylabel('y (m)')
plt.show()

plt.figure()
plt.plot(t_result,x_result[:,2])
plt.xlabel('Time (s)')
plt.ylabel('z (m)')
plt.show()

plt.figure()
plt.plot(t_result,u_result[:,0])
plt.xlabel('Time (s)')
plt.ylabel('Thrust (N)')
plt.show()

plt.figure()
plt.plot(t_result,u_result[:,1], label='tx')
plt.plot(t_result,u_result[:,2], label='ty')
plt.plot(t_result,u_result[:,3], label='tz')
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.show()

plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(x_result[:,0],x_result[:,1],x_result[:,2],'c-',label='Trajectory')
ax.plot3D(xtarg[0],xtarg[1],xtarg[2],'o',label='Target')
plt.legend()
ax.set_xlabel('x, m')
ax.set_ylabel('y, m')
ax.set_zlabel('z, m')
plt.show()



