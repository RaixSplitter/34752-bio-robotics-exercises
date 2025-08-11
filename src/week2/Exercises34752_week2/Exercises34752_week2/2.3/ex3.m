clear all;
close all;

% Initialization
% Mass of the arm
m = 1;
% Length of the arm
arm_length = 0.3;
% Inertia
Inertia = 1/3 * m *(arm_length)^2;
% Gravity
g = 9.81;
% Damping coefficient for the forward model
dampCoeff = 0.5;

% Sensory delay (s)
delay = 0;
% Simulation time step (s)
dt = 0.001;
% Simulation duration (s)
max_time = 10;

% Time to start the movement (s)
mvt_start_time = 0.5;

% Target joint angle
final_target_ang = 2*pi/2;

% Perturbation start time (s)
pert_start_time = .75;
% Perturbation end time (s)
pert_end_time = 1;
% Perturbation amplitude
pert_amp = 1;

% Gain and damping parameters for forward model
Kp_forward = 0;
Kd_forward = 0;

% Gain and damping parameters for feedback model
Kp_feedback = 10;
Kd_feedback = 0.5;

%% Run the simulation

sim("simulink_model.slx")

%% Plot results
% plot(ans);


