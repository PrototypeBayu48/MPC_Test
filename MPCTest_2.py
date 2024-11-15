import numpy as np
import sys
from casadi import *

import do_mpc

import matplotlib.pyplot as plt

from matplotlib import rcParams

import time

mass = 2.0
gravity = 9.81
l = 0.5
Ixx = 0.05

dt = 0.02

xtarg = np.array([10.0, 10.0, 0.0, 0.0, 0.0, 0.0])

model_type = 'continuous' # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)

y = model.set_variable(var_type='_x', var_name='y', shape=(1,1))
z = model.set_variable(var_type='_x', var_name='z', shape=(1,1))
theta = model.set_variable(var_type='_x', var_name='theta', shape=(1,1))
dy = model.set_variable(var_type='_x', var_name='dy', shape=(1,1))
dz = model.set_variable(var_type='_x', var_name='dz', shape=(1,1))
dtheta = model.set_variable(var_type='_x', var_name='dtheta', shape=(1,1))

T1 = model.set_variable(var_type='_u', var_name='T1')
T2 = model.set_variable(var_type='_u', var_name='T2')

model.set_rhs('y', dy)
model.set_rhs('z', dz)
model.set_rhs('theta', dtheta)

dy_next = -(T1/mass)*sin(theta) - (T2/mass)*sin(theta)
dz_next = -gravity + (T1/mass)*cos(theta) + (T2/mass)*cos(theta)
dtheta_next = -(l*T1)/Ixx + (l*T2)/Ixx
model.set_rhs('dy', dy_next)
model.set_rhs('dz', dz_next)
model.set_rhs('dtheta', dtheta_next)

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

costpen = np.array([10.0, 10.0, 1.0, 0.0, 0.0, 0.0])

cost = costpen[0]*(y - xtarg[0])**2 + costpen[1]*(z - xtarg[1])**2 + costpen[2]*(theta - xtarg[2])**2 + costpen[3]*(dy - xtarg[3])**2 + costpen[4]*(dz - xtarg[4])**2 + costpen[5]*(dtheta - xtarg[5])**2

mpc.set_objective(mterm=0*cost, lterm=cost)
mpc.set_rterm(T1=0.1, T2=0.1) # penalty on input changes

# lower bounds of the states
mpc.bounds['lower', '_x', 'y'] = -20.0
mpc.bounds['lower', '_x', 'z'] = 0.0
mpc.bounds['lower', '_x', 'theta'] = -2*np.pi
mpc.bounds['lower', '_x', 'dy'] = -20.0
mpc.bounds['lower', '_x', 'dz'] = -20.0
mpc.bounds['lower', '_x', 'dtheta'] = -20.0

# upper bounds of the states
mpc.bounds['upper', '_x', 'y'] = 20.0
mpc.bounds['upper', '_x', 'z'] = 20.0
mpc.bounds['upper', '_x', 'theta'] = 2*np.pi
mpc.bounds['upper', '_x', 'dy'] = 20.0
mpc.bounds['upper', '_x', 'dz'] = 20.0
mpc.bounds['upper', '_x', 'dtheta'] = 20.0

# upper and lower bounds of the control input
mpc.bounds['lower','_u','T1'] = 0.0
mpc.bounds['lower','_u','T2'] = 0.0

mpc.bounds['upper','_u','T1'] = mass*gravity
mpc.bounds['upper','_u','T2'] = mass*gravity

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
x0 = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
u0 = np.array([mass*gravity/2, mass*gravity/2])

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
    dist = np.linalg.norm(x0[:2] - xtarg[:2])
    if x0[1] < 0.0:
        break
    if dist < 0.1:
        break

results = simulator.data
t_result = results['_time']
x_result = results['_x']
u_result = results['_u']

print('Average Computation Time: ', np.mean(tcomp_result))
print('Max Computation Time: ', np.max(tcomp_result))
print('Min Computation Time: ', np.min(tcomp_result))
print('Computation Time Standard Deviation: ', np.std(tcomp_result))

plt.figure()
plt.plot(t_result,x_result[:,0])
plt.xlabel('Time (s)')
plt.ylabel('y (m)')
plt.show()

plt.figure()
plt.plot(t_result,x_result[:,1])
plt.xlabel('Time (s)')
plt.ylabel('z (m)')
plt.show()

plt.figure()
plt.plot(t_result,x_result[:,2])
plt.xlabel('Time (s)')
plt.ylabel('Theta (rad)')
plt.show()

plt.figure()
plt.plot(t_result,u_result[:,0], label='T1')
plt.plot(t_result,u_result[:,1], label='T2')
plt.xlabel('Time (s)')
plt.ylabel('Thrust (N)')
plt.legend()
plt.show()

plt.figure()
plt.plot(x_result[:,0],x_result[:,1], label='Trajectory')
plt.plot(xtarg[0],xtarg[1],'o', label='Target')
plt.xlabel('y (m)')
plt.ylabel('z (m)')
plt.legend()
plt.show()



