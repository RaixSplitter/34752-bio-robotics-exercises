import numpy as np
import matplotlib.pyplot as plt

from adaptive_filter.cerebellum import AdaptiveFilterCerebellum
from robot import SingleLink
from tqdm import tqdm

Ts = 1e-3
n_inputs = 1
n_outputs = 1
n_bases = 4
beta = .00001

c = AdaptiveFilterCerebellum(Ts, n_inputs, n_outputs, n_bases, beta)

T_end = 10 # in one trial
n_steps = int(T_end/Ts) # in one trial
n_trials = 30

plant = SingleLink(Ts)

## Logging variables
t_vec = np.array([Ts*i for i in range(n_steps*n_trials)])

theta_vec = np.zeros(n_steps*n_trials)
theta_ref_vec = np.zeros(n_steps*n_trials)

## Feedback controller variables
Kp = 30
Kv = 0

## TODO: Define parameters for periodic reference trajectory
# xmin = [-2*np.pi, -2*np.pi]
# xmax = [2*np.pi, 2*np.pi]
xmin = [0, 0]
xmax = [1, 1]

## TODO: CMAC initialization
n_rfs = 15


T = 5
C_t = 0
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
    error_fb = error + C_t
    tau_m = Kp * error_fb + Kv * (-omega)

    ## TODO: Implement the CMAC controller into the loop
    
    C_t = c.step(tau_m, error)
    
    # print(f"Error {error}, Tau {tau_m}, {theta}")
    # Iterate simulation dynamics
    plant.step(tau_m)

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

## TODO: Change the code to the recurrent architecture
# You can update the cerebellum with: C = c.step(u, error)

## TODO: Plot results