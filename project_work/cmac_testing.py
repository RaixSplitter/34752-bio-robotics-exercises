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
T_end = 1 # in one trial
n_steps = int(T_end/Ts) # in one trial
n_trials = 10

# data = np.loadtxt("data.csv", delimiter=",")
# end_pos = data[:,2:].T
# x = torch.from_numpy(end_pos.T).float()
# data_mean = x.mean(axis=0)
data_mean = np.array([320, 390])
# plant = SingleLink(Ts)

## Logging variables
t_vec = np.array([Ts*i for i in range(n_steps*n_trials)])

theta_vec = np.zeros(n_steps*n_trials)
theta_ref_vec = np.zeros(n_steps*n_trials)

## Feedback controller variables
Kp = 30
Kv = 0

## Define parameters for periodic reference trajectory
xmin = np.array([150, 350]) - data_mean
xmax = np.array([490, 430]) - data_mean

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

    def random_point(self):
        if not self.new:
            x = np.random.randint(150,490)
            y = np.random.randint(350,430)
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
timer = 0
wait_for_update = False
wait_for_confirm = False
active_run = True

## Simulation loop
for i in range(n_steps*n_trials):
    while active_run:
        frame = ct.capture_image(cam)
        # frame = cv2.flip(frame, 1)
        coordinateStore1.random_point()
        cv2.circle(frame,coordinateStore1.point,3,(255,0,0),-1)
        cv2.imshow("live_cam", frame)
        k = cv2.waitKey(10)
        
        # if coordinateStore1.new:
        #     print(coordinateStore1.point)
        #     coordinateStore1.new = False
        if k == 27: # escape
            break
        
        if wait_for_confirm:
            if k == 32: # space bar
                break
        elif wait_for_update: # wait for robot to move to position
            if (datetime.now()-timer).seconds >= 2:
                frame = ct.capture_image(cam)
                new_theta = ct.locate(frame) # (x, y)
                if new_theta[0] == None:
                    print("Brick not in view")
                    break
                new_theta -= data_mean
                error = np.sum((new_theta - theta_ref)**2) # calc error (dist)
                # print(c.w[0,:])
                c.learn(tau_MLP)
                # print(c.w[0,:])
                wait_for_confirm = True
        
        elif coordinateStore1.new: # point is available
            # Measure
            frame = ct.capture_image(cam)
            theta = ct.locate(frame) # (x, y)
            if theta[0] == None:
                print("Brick not in view")
                break
            theta_ref  = np.array(coordinateStore1.point, dtype=np.float64)
            
            theta     -= data_mean
            theta_ref -= data_mean
            
            # MLP
            with torch.no_grad():
                outp = model(torch.from_numpy(theta_ref).float())
                tau_MLP = outp.numpy()
                print("MLP:",tau_MLP)
            # cmac
            tau_cmac = c.predict([theta, theta_ref])
            tau = tau_MLP + tau_cmac
            print("Same as previous", (tau != tau_MLP).any())
            
            # Iterate simulation dynamics
            # plant.step(tau)
            api.setPos(tau[0], tau[1], module)
            
            # log
            # theta_vec[i] = plant.theta
            # theta_ref_vec[i] = theta_ref
            
            timer = datetime.now()
            wait_for_update = True
            
    coordinateStore1.new = False
    wait_for_confirm = False
    wait_for_update = False
    if k == 27: # escape
        break

# ## Plotting
# plt.plot(t_vec, theta_vec, label='theta')
# plt.plot(t_vec, theta_ref_vec, '--', label='reference')
# plt.legend()

# # Plot trial error
# error_vec = theta_ref_vec - theta_vec
# l = int(T/Ts)
# trial_error = np.zeros(n_trials)
# for t in range(n_trials):
#    trial_error[t] = np.sqrt( np.mean( error_vec[t*l:(t+1)*l]**2 ) )
# plt.figure()
# plt.plot(trial_error)

# plt.show()

print('Terminating')
api.terminate()

