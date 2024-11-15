import numpy as np
import sys
from casadi import *

import do_mpc

import matplotlib.pyplot as plt

from matplotlib import rcParams

import time

from scipy.spatial.transform import Rotation as R

def Rmat(q):
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

n = 13
m = 4

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

dt = 0.02

xtarg = np.zeros(n)

xtarg[0] = 5.0
xtarg[1] = 5.0
xtarg[2] = 5.0

rtarg_euler = R.from_euler('xyz', [0.0, 0.0, 0.0])
rtarg_quat = rtarg_euler.as_quat(scalar_first = True)

xtarg[3:7] = np.array(rtarg_quat)

model_type = 'continuous' # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)

x = model.set_variable(var_type='_x', var_name='x', shape=(1,1))
y = model.set_variable(var_type='_x', var_name='y', shape=(1,1))
z = model.set_variable(var_type='_x', var_name='z', shape=(1,1))
q_w = model.set_variable(var_type='_x', var_name='q_w', shape=(1,1))
q_x = model.set_variable(var_type='_x', var_name='q_x', shape=(1,1))
q_y = model.set_variable(var_type='_x', var_name='q_y', shape=(1,1))
q_z = model.set_variable(var_type='_x', var_name='q_z', shape=(1,1))

dx = model.set_variable(var_type='_x', var_name='dx', shape=(1,1))
dy = model.set_variable(var_type='_x', var_name='dy', shape=(1,1))
dz = model.set_variable(var_type='_x', var_name='dz', shape=(1,1))
omega_x = model.set_variable(var_type='_x', var_name='omega_x', shape=(1,1))
omega_y = model.set_variable(var_type='_x', var_name='omega_y', shape=(1,1))
omega_z = model.set_variable(var_type='_x', var_name='omega_z', shape=(1,1))

Th = model.set_variable(var_type='_u', var_name='Th')
tx = model.set_variable(var_type='_u', var_name='tx')
ty = model.set_variable(var_type='_u', var_name='ty')
tz = model.set_variable(var_type='_u', var_name='tz')

model.set_rhs('x', dx)
model.set_rhs('y', dy)
model.set_rhs('z', dz)

q_w_next = 0.5*(-omega_x*q_x - omega_y*q_y - omega_z*q_z)
q_x_next = 0.5*(omega_x*q_w + omega_y*q_z - omega_z*q_y)
q_y_next = 0.5*(-omega_x*q_z + omega_y*q_w + omega_z*q_x)
q_z_next = 0.5*(omega_x*q_y - omega_y*q_x + omega_z*q_w)

model.set_rhs('q_w', q_w_next)
model.set_rhs('q_x', q_x_next)
model.set_rhs('q_y', q_y_next)
model.set_rhs('q_z', q_z_next)

dx_next = -2*(q_x*q_z + q_w*q_y)*gravity
dy_next = -2*(q_y*q_z - q_w*q_x)*gravity
dz_next = (Th/mass) - (1 - 2*((q_x**2) + (q_y**2)))*gravity

model.set_rhs('dx', dx_next)
model.set_rhs('dy', dy_next)
model.set_rhs('dz', dz_next)

omega_x_next = omega_y*omega_z*c1 + (tx/Ixx)
omega_y_next = omega_x*omega_z*c2 + (ty/Iyy)
omega_z_next = omega_x*omega_y*c3 + (tz/Izz)

model.set_rhs('omega_x', omega_x_next)
model.set_rhs('omega_y', omega_y_next)
model.set_rhs('omega_z', omega_z_next)

# Build the model
model.setup()

mpc = do_mpc.controller.MPC(model)

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

xtarg_body = xtarg
xtarg_body[:3] = Rmat([q_w, q_x, q_y, q_z])@xtarg_body[:3]

#cost = costpen[0]*(x - xtarg_body[0])**2 + costpen[1]*(y - xtarg_body[1])**2 + costpen[2]*(z - xtarg_body[2])**2
cost = (x - 5.0)**2 + (y - 5.0)**2 + (z - 5.0)**2

mpc.set_objective(mterm=0*cost, lterm=cost)
mpc.set_rterm(Th=0.1, tx=1.0, ty=1.0, tz=1.0) # penalty on input changes

# lower bounds of the states
mpc.bounds['lower', '_x', 'x'] = -20.0
mpc.bounds['lower', '_x', 'y'] = -20.0
mpc.bounds['lower', '_x', 'z'] = 0.0

# upper bounds of the states
mpc.bounds['upper', '_x', 'x'] = 20.0
mpc.bounds['upper', '_x', 'y'] = 20.0
mpc.bounds['upper', '_x', 'z'] = 20.0

# lower bounds of the input
mpc.bounds['lower','_u','Th'] = 0.0
mpc.bounds['lower','_u','tx'] = -max_tx
mpc.bounds['lower','_u','ty'] = -max_ty
mpc.bounds['lower','_u','tz'] = -max_tz

# upper bounds of the input
mpc.bounds['upper','_u','Th'] = max_thrust
mpc.bounds['upper','_u','tx'] = max_tx
mpc.bounds['upper','_u','ty'] = max_ty
mpc.bounds['upper','_u','tz'] = max_tz

mpc.setup()

simulator = do_mpc.simulator.Simulator(model)

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
x_result_body = results['_x']
u_result = results['_u']

x_result = x_result_body

for n1 in range(len(np.squeeze(t_result))):
    x_result[n1,:3] = np.transpose(Rmat(x_result[n1,3:7]))@x_result_body[n1,:3]
    x_result[n1,7:10] = np.transpose(Rmat(x_result[n1,3:7]))@x_result_body[n1,7:10]

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



