import numpy as np
import matplotlib.pyplot as plt

from robot import SingleLink
from cmac2 import CMAC
from tqdm import tqdm
## Initialize simulation
Ts = 1e-2
T_end = 10 # in one trial
n_steps = int(T_end/Ts) # in one trial
n_trials = 100

plant = SingleLink(Ts)

## Logging variables
t_vec = np.array([Ts*i for i in range(n_steps*n_trials)])

theta_vec = np.zeros(n_steps*n_trials)
theta_ref_vec = np.zeros(n_steps*n_trials)

## Feedback controller variables
Kp = 30
Kv = 0

## TODO: Define parameters for periodic reference trajectory
xmin = [-2*np.pi, -2*np.pi]
xmax = [2*np.pi, 2*np.pi]
# xmin = [0, 0]
# xmax = [1, 1]

## TODO: CMAC initialization
n_rfs = 15
c = CMAC(n_rfs, xmin, xmax, .1)

T = 5
## Simulation loop
for i in tqdm(range(n_steps*n_trials)):

    t = i*Ts
    ## TODO: Calculate the reference at this time step
    # theta_ref = np.pi/4
    theta_ref = np.pi * np.sin(2* np.pi * t/T)
    
    # Measure
    theta = plant.theta
    omega = plant.omega

    # Feedback controler
    error = (theta_ref - theta)
    tau_m = Kp * error + Kv * (-omega)

    ## TODO: Implement the CMAC controller into the loop
    tau_cmac = c.predict([theta, theta_ref])
    c.learn(tau_m)
    tau = tau_m + tau_cmac
    
    # Iterate simulation dynamics
    plant.step(tau)

    theta_vec[i] = plant.theta
    theta_ref_vec[i] = theta_ref

## Plotting
plt.plot(t_vec, theta_vec, label='theta')
plt.plot(t_vec, theta_ref_vec, '--', label='reference')
plt.legend()

# Plot trial error
error_vec = theta_ref_vec - theta_vec
l = int(T/Ts)
trial_error = np.zeros(n_trials)
for t in range(n_trials):
   trial_error[t] = np.sqrt( np.mean( error_vec[t*l:(t+1)*l]**2 ) )
plt.figure()
plt.plot(trial_error)

plt.show()