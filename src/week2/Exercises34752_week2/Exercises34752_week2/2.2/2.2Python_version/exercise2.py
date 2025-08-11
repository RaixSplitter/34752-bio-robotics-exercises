import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
# from numpy.core.fromnumeric import size
from SimFunctions import SimulationFunctions

## Parameters
NOISE = 0.0
DELAY = 1
# Movement duration
T = 0.6
# Time step
dt = 0.01
# Simulation duration
L = 6.0
# Proportional parameter
kp = 400.0
# Derivative parameter
kd = 11
# Upper arm length
le1 = 0.3
# Lower arm length
le2 = 0.3
# Upper arm mass
m1 = 3
# Lower arm mass
m2 = 3
# Gravity
g = -9.8

## Functions
Var = [T, dt, L, kp, kd, le1, le2, m1, m2, g]
Sim = SimulationFunctions(Var)

## Variables
# Joint angles [shoulder elbow]  [rad]
ang = [-np.pi / 4, np.pi]
ang_rec = np.zeros((int(L / dt + 1), 2))
delayed_ang = ang[:]

# Joint velocity [shoulder elbow] [rad/s]
vel = [0, 0]
vel_rec = np.zeros((int(L / dt + 1), 2))
delayed_vel = vel[:]

# Joint acceleration [shoulder elbow]  [rad/s^2]
acc = [0, 0]
acc_rec = np.zeros((int(L / dt + 1), 2))

# Jerk [shoulder elbow]
jerk_rec = np.zeros((int(L / dt + 1), 2))

# Shoulder position
shoulder_pos = [0, 0]
# Elbow position
elbow_pos_rec = np.zeros((int(L / dt) + 1, 2))
# Wrist position
wrist_pos = [0, 0]
wrist_pos_rec = np.zeros((int(L / dt + 1), 2))

# Initial wrist position for current movement
init_wrist_pos = wrist_pos
# Desired wrist position
final_wrist_pos = [
    [0.3, 0.0],
    [0.0, 0.0],
    [0.3 * np.cos(np.pi / 4), 0.3 * np.sin(np.pi / 4)],
    [0.0, 0.0],
    [0.0, 0.3],
    [0.0, 0.0],
    [0.3 * np.cos(3 * np.pi / 4), 0.3 * np.sin(3 * np.pi / 4)],
    [0.0, 0.0],
]

# Current target index
curr_target = 0
# Movement start_time
start_t = 0

# TODO define time steps of delay
# Define delay buffer for angles and velocities
delay_steps = DELAY  # Number of time steps to delay
ang_buffer = np.zeros((delay_steps + 1, 2))  # Buffer to store past angles
vel_buffer = np.zeros((delay_steps + 1, 2))  # Buffer to store past velocities

# Initialize buffers with current values
for i in range(delay_steps + 1):
    ang_buffer[i, :] = ang
    vel_buffer[i, :] = vel


## Simulation - plot setup

## Reset variables
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_xlabel("meters", fontsize=10)
ax.set_ylabel("meters", fontsize=10)
ax.set_xlim([-0.5, 0.5])
ax.set_ylim([-0.5, 0.5])

Time = time.time()
for t in np.arange(0, int(L), dt):
    # Update records
    ang_rec[round(t / dt) + 1, :] = ang
    vel_rec[round(t / dt) + 1, :] = vel
    acc_rec[round(t / dt) + 1, :] = acc
    if t > 0:
        jerk_rec[round(t / dt) + 1, :] = acc - acc_rec[round(t / dt), :]

    ## Current wrist target
    current_wrist_target = final_wrist_pos[curr_target][:]

    if curr_target <= 7:
        ## Planner
        # Get desired position from planner
        if t - start_t < T:
            desired_pos = Sim.minjerk(init_wrist_pos, current_wrist_target, t - start_t)

        ## Inverse kinematics
        # Get desired angle from inverse kinematics
        desired_ang = np.real(Sim.invkinematics(desired_pos)).ravel()

        ## Inverse dynamics
        ## TODO Define delayed angles and velocities
        # Update delay buffers - shift old values and add new ones
        ang_buffer[1:, :] = ang_buffer[:-1, :]  # Shift buffer
        vel_buffer[1:, :] = vel_buffer[:-1, :]  # Shift buffer
        ang_buffer[0, :] = ang  # Add current angle to buffer
        vel_buffer[0, :] = vel  # Add current velocity to buffer
        
        # Get delayed values from buffer
        delayed_ang = ang_buffer[delay_steps, :].copy()  # Get delayed angle
        delayed_vel = vel_buffer[delay_steps, :].copy()  # Get delayed velocity

        ## TODO Compute torque with delayed angles and velocities

        # Get desired torque from PD controller
        desired_torque = Sim.pdcontroller(desired_ang, delayed_ang, delayed_vel)

        ## Forward dynamics
        ## DEFINE NOISE - you can use randn
        noise = (
            np.random.randn(*np.shape(desired_torque)) * NOISE
        )  # Adjust noise scale as needed
        ## ADD NOISE to torque
        desired_torque = desired_torque + noise
        # Pass torque to plant
        ang, vel, acc = Sim.plant(delayed_ang, delayed_vel, acc, desired_torque)

        ## Forward kinematics
        # Calculate new joint positions
        elbow_pos, wrist_pos = Sim.fkinematics(ang)
        elbow_pos_rec[round(t / dt) + 1, :] = elbow_pos
        wrist_pos_rec[round(t / dt) + 1, :] = wrist_pos

        # Record wrist position
        wrist_pos_rec[round(t / dt) + 1, :] = wrist_pos

        ## Next target
        if (t - start_t >= T + 0.02) & (curr_target < 7):
            curr_target = curr_target + 1
            init_wrist_pos = wrist_pos
            start_t = t

# Plot arm, wrist path, and targets -- ANIMATION
if animation:
    ax.scatter(
        np.array(final_wrist_pos)[:, 0],
        np.array(final_wrist_pos)[:, 1],
        color="green",
        label="Targets",
    )

    for t2 in np.arange(0, t, dt):
        ax.cla()
        ax.scatter(
            np.array(final_wrist_pos)[:, 0],
            np.array(final_wrist_pos)[:, 1],
            color="green",
            label="Targets",
        )

        shoulder_x, shoulder_y = shoulder_pos[0], shoulder_pos[1]
        elbow_x, elbow_y = (
            elbow_pos_rec[int(t2 / dt), 0],
            elbow_pos_rec[int(t2 / dt), 1],
        )
        wrist_x, wrist_y = (
            wrist_pos_rec[int(t2 / dt), 0],
            wrist_pos_rec[int(t2 / dt), 1],
        )

        arm_x = [shoulder_x, elbow_x, wrist_x]
        arm_y = [shoulder_y, elbow_y, wrist_y]
        ax.plot(arm_x, arm_y, color="blue", label="Arm")

        ax.plot(
            wrist_pos_rec[: int(t2 / dt) + 1, 0],
            wrist_pos_rec[: int(t2 / dt) + 1, 1],
            "--",
            color="red",
            linewidth=0.5,
            label="Wrist Path",
        )

        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_xlabel("Meters")
        ax.set_ylabel("Meters")

        plt.pause(0.001)

elapsed = time.time() - Time
print("Time elapsed:", elapsed)

ax.plot(wrist_pos_rec[:-1, 0], wrist_pos_rec[:-1, 1], "--", color="red", linewidth=0.5)
ax.scatter(
    np.array(final_wrist_pos)[:, 0], np.array(final_wrist_pos)[:, 1], color="green"
)
plt.show()

plt.subplot(3, 1, 1)
[A, B] = plt.plot(
    np.arange(0, L - dt, dt),
    [xx[0] for xx in vel_rec[: int(L / dt) - 1]],
    np.arange(0, L - dt, dt),
    [xx[1] for xx in vel_rec[: int(L / dt) - 1]],
)
plt.legend([A, B], ["Shoulder", "Elbow"])
plt.xlabel("time (ms)")
plt.ylabel("velocity")
plt.subplot(3, 1, 2)
[A, B] = plt.plot(
    np.arange(0, L - dt, dt),
    [xx[0] for xx in acc_rec[: int(L / dt) - 1]],
    np.arange(0, L - dt, dt),
    [xx[1] for xx in acc_rec[: int(L / dt) - 1]],
)
plt.legend([A, B], ["Shoulder", "Elbow"])
plt.xlabel("time (ms)")
plt.ylabel("acceleration")
plt.subplot(3, 1, 3)
[A, B] = plt.plot(
    np.arange(0, L - dt, dt),
    [xx[0] for xx in jerk_rec[: int(L / dt) - 1]],
    np.arange(0, L - dt, dt),
    [xx[1] for xx in jerk_rec[: int(L / dt) - 1]],
)
plt.legend([A, B], ["Shoulder", "Elbow"])
plt.xlabel("time (ms)")
plt.ylabel("jerk")
plt.tight_layout()
plt.show()
