import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from datetime import datetime

from utils.cmac2 import CMAC
from utils.robot import SingleLink
import utils.torch_model as torch_model
import utils.camera_tools as ct
from FableAPI.fable_init import api

from tqdm import tqdm

## Initialize simulation
Ts = 1e-2
T_end = 10 # in one trial
n_steps = int(T_end/Ts) # in one trial
n_trials = 100

# data = np.loadtxt("data.csv", delimiter=",")
# end_pos = data[:,2:].T
# x = torch.from_numpy(end_pos.T).float()
# data_mean = x.mean(axis=0)
data_mean = np.array([236.8800, 342.2700])
# plant = SingleLink(Ts)

## Logging variables
t_vec = np.array([Ts*i for i in range(n_steps*n_trials)])

theta_vec = np.zeros(n_steps*n_trials)
theta_ref_vec = np.zeros(n_steps*n_trials)

## Feedback controller variables
Kp = 30
Kv = 0

## Define parameters for periodic reference trajectory
xmin = [-2*np.pi, -2*np.pi]
xmax = [2*np.pi, 2*np.pi]

## CMAC initialization
n_rfs = 15
c = CMAC(n_rfs, xmin, xmax, 2, .1)

T = 5

## Camera + robot initialization
# Initialize the camera first.. waits for it to detect the green block
def initialize_camera(cam):
    while True:
        frame = ct.capture_image(cam)
        x, _ = ct.locate(frame)

        if x is not None:
            break

# Initialize the robot module
def initialize_robot(module=None):
    api.setup(blocking=True)
    # Find all the robots and return their IDs.
    print('Search for modules')
    moduleids = api.discoverModules()

    if module is None:
        module = moduleids[0]
    print('Found modules: ',moduleids)
    api.setPos(0,0, module)
    api.sleep(0.5)

    return module

# dummy class for targets
class CoordinateStore:
    def __init__(self):
        self.point = None
        self.new = False

    def select_point(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDBLCLK and not self.new:
            cv2.circle(frame,(x,y),3,(255,0,0),-1)
            self.point = [x,y]
            self.new = True

cam = ct.prepare_camera()
assert cam.isOpened(), "No camera"  # False
initialize_camera(cam)
module = initialize_robot()

# Set move speed
speedX = 25
speedY = 25
api.setSpeed(speedX, speedY, module)

# Set accuracy
accurateX = 'HIGH'
accurateY = 'HIGH'
api.setAccurate(accurateX, accurateY, module)

# Load the trained model
model = torch_model.MLPNet(2, 24, 2)
model.load_state_dict(torch.load('650_trained_model.pth', weights_only=True))

# Instantiate class
coordinateStore1 = CoordinateStore()

cv2.namedWindow("live_cam")
cv2.setMouseCallback('live_cam', coordinateStore1.select_point)
using_cmac = False
theta_ref = []
feedforward_t = []
timer = 0
steps = 0
faster = 0

## Simulation loop
for i in tqdm(range(n_trials)):
    while steps <= n_steps:
        frame = ct.capture_image(cam)
        # frame = cv2.flip(frame, 1)
        cv2.imshow("live_cam", frame)
        k = cv2.waitKey(10)
        if k == 27:
            break
        
        if using_cmac:
            if (datetime.now()-timer).seconds + faster >= 2:
                # Measure
                # theta = plant.theta
                # omega = plant.omega
                frame = ct.capture_image(cam)
                theta = ct.locate(frame) # (x, y)

                ## Implement the CMAC controller into the loop
                tau_cmac = c.predict([theta, theta_ref])
                print(c.w[:,0,0])
                c.learn(feedforward_t)
                tau = feedforward_t + tau_cmac
                print(tau)
                
                # Iterate simulation dynamics
                # plant.step(tau)
                api.setPos(tau[0], tau[1], module)
                feedforward_t = tau
                
                # theta_vec[i] = plant.theta
                # theta_ref_vec[i] = theta_ref
                steps += 1
                timer = datetime.now()
                faster = 1.5
                if (steps)%10==0:
                    print("Steps:",steps)
            
        
        elif coordinateStore1.new: # point is available
            theta_ref  = np.array(coordinateStore1.point, dtype=np.float64)
            theta_ref -= data_mean
            
            # using non-active learning model prediction
            with torch.no_grad():
                outp = model(torch.from_numpy(theta_ref).float())
                t = outp.numpy()
                print("MLP:",t)
            feedforward_t = t
            api.setPos(t[0], t[1], module)
            timer = datetime.now()
            using_cmac = True
            # coordinateStore1.new = False
    steps = 0
    using_cmac = False
    coordinateStore1.new = False
    if k == 27:
        break

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

print('Terminating')
api.terminate()

