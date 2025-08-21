import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from datetime import datetime

from utils.cerebellum import AdaptiveFilterCerebellum
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


## CMAC initialization
n_rfs = 15
# c = CMAC(n_rfs, xmin, xmax, 2, .1)
Ts = 1e-3
n_inputs = 2
n_outputs = 2
n_bases = 4
beta = .00001
c = AdaptiveFilterCerebellum(Ts, n_inputs, n_outputs, n_bases, beta)

T = 5
MOVE_STEP = 100
AF_THRESHHOLD = 50


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
            # x = np.random.randint(150,490) ## Sonnys
            # y = np.random.randint(350,430)
            # x = np.random.randint(380,820) ## Simons
            # y = np.random.randint(250,620)
            
            x = np.random.randint(235,485) # Markus
            y = np.random.randint(145,280)
            


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
model = torch_model.MLPNet(2, 24, 16, 8, 2)
model.load_state_dict(torch.load('trained_model_diff.pth', weights_only=True))

# Instantiate class
coordinateStore1 = CoordinateStore()

cv2.namedWindow("live_cam")
cv2.setMouseCallback('live_cam', coordinateStore1.select_point)
timer = 0
wait_for_update     = False
wait_for_confirm    = False
active_run          = True
error               = np.array([0,0])


## Simulation loop
for i in range(n_steps*n_trials):
    # new_theta           = None
    tau_MLP             = [None,None]
    move_step_counter = 0
    p_id = 0 #point id
    adaptive_filter_toggle = False
    while active_run:
        move_step_counter += 1
        if move_step_counter >= MOVE_STEP:
            p_id += 1 #point id
            coordinateStore1.new = False
            wait_for_confirm = False
            wait_for_update = False
            move_step_counter = 0
            if p_id > AF_THRESHHOLD:
                adaptive_filter_toggle = True


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
                theta_new = None
                break
        elif wait_for_update: # wait for robot to move to position
            if (datetime.now()-timer).seconds >= 0.1:
                frame = ct.capture_image(cam)
                new_theta = ct.locate(frame) # (x, y)
                if new_theta[0] == None:
                    print("Brick not in view")
                    break
                # new_theta = (np.array(new_theta) + 750)/1500.
               
                dist = theta_ref - new_theta
                with open("run_log_af_toggle_test.csv", "a") as f:
                    timestamp = datetime.now().isoformat()
                    f.write(f"{p_id},{adaptive_filter_toggle},{dist[0]},{dist[1]},{timestamp}\n") # p_id, adaptive_filter_toggle, dist_x, dist_y, timestamp

                error       = (dist+ 750)/1500. # calc error (dist)
                
                # print(c.w[0,:])
                # c.learn(tau_MLP)
                # print(c.w[0,:])
                # wait_for_confirm = True
                wait_for_update = False
                # if np.linalg.norm(error) <= 0.65:
                #     print(error)
                #     break
            

        
        elif coordinateStore1.new: # point is available
            # Measure
            frame = ct.capture_image(cam)
            theta = ct.locate(frame) # (x, y)
            if theta[0] == None:
                print("Brick not in view")
                break
            theta_ref  = np.array(coordinateStore1.point, dtype=np.float64)
            
            # theta     = (np.array(theta) + 750.)/1500.
            # theta_ref = (np.array(theta_ref) + 750.)/1500.
            error       = (( theta_ref - theta)+750)/1500 # calc error (dist)       
            if tau_MLP[0] != None:
                C_t         = c.step(tau_MLP, error)
                if adaptive_filter_toggle:
                    error    += C_t
            # MLP
            with torch.no_grad():
                # tmeas1 = api.getPos(0,module)
                # tmeas2 = api.getPos(1,module)
                print(error)
                outp = model(torch.tensor(error).float())
                tau_MLP = outp.numpy()
                
                print("MLP:",tau_MLP)
                print("Current position: ", theta)
                print("Reference: ", theta_ref)
            tau = tau_MLP
            
            # Iterate simulation dynamics
            # plant.step(tau)
            t0 = api.getPos(0,module)
            t1 = api.getPos(1,module)
            print("\nAPI")
            print("t0,t1",t0,t1)
            print("tau",tau)
            print("set_pos",t0+tau[0], t1+tau[1])
            res = api.setPos(t0+tau[0], t1+tau[1], module)
            print(res)
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

