import numpy as np
import camera_tools as ct
import cv2
import datetime
from FableAPI.fable_init import api

# We have provided camera_tools as a stand-alone python file in ~/camera_tools/camera_tools.py
# The same functions are available in the Conda 'biocontrol' venv that is provided

cam = ct.prepare_camera()
print(cam.isOpened())
print(cam.read())

i = 0

# Initialization of the camera. Wait for sensible stuff
def initialize_camera(cam):
    while True:
        frame = ct.capture_image(cam)

        x, _ = ct.locate(frame)

        if x is not None:
            break     

def initialize_robot(module=None):
    api.setup(blocking=True)
    # Find all the robots and return their IDs
    print('Search for modules')
    moduleids = api.discoverModules()

    if module is None:
        module = moduleids[0]
    print('Found modules: ',moduleids)
    api.setPos(0,0, module)
    api.sleep(0.5)
    return module

def get_r_angle():
    t1 = np.random.uniform(-85,0)
    t2 = np.random.uniform(-85,86)
    return (t1,t2)

initialize_camera(cam)
module = initialize_robot()

# Write DATA COLLECTION part - the following is a dummy code
# Remember to check the battery level and to calibrate the camera
# Some important steps are: 
# 1. Define an input/workspace for the robot; 
# 2. Collect robot data and target data
# 3. Save the data needed for the training

n_t1 = 20
n_t2 = 10
num_datapoints = n_t1*n_t2

# uniform change
# t1 = np.repeat(np.linspace(-85, 0, n_t1), n_t2) # repeat the vector
# t2 = np.tile(np.linspace(-85, 86, n_t2), n_t1) # repeat each element
# thetas = np.stack((t1,t2))
# api.setPos(thetas[0,i], thetas[1,i], module)

# random
api.setPos(*get_r_angle(), module)

class TestClass:
    def __init__(self, num_datapoints):
        self.i = 0
        self.num_datapoints = num_datapoints
        self.data = np.zeros( (num_datapoints, 4) )
        # self.data = np.zeros( (num_datapoints, 6) ) # for angle
        self.time_of_move = datetime.datetime.now()
        self.timer = 0.5
        self.xy = [None,None]
        self.t  = [None,None]

    def go(self):
        while True:
            if self.xy[0] is None:
                img       = ct.capture_image(cam)
                self.xy   = ct.locate(img)
                self.t[0] = api.getPos(0,module)
                self.t[1] = api.getPos(1,module)
            if self.i >= num_datapoints:
                np.savetxt("data.csv", self.data, delimiter=",")
                return True
            frame = ct.capture_image(cam)
            cv2.imshow("test", frame)
            k = cv2.waitKey(10)
            if k == 27:
                break
            if (datetime.datetime.now() - self.time_of_move).total_seconds() > self.timer and not (api.getMoving(0,module) or api.getMoving(1,module)):
                img = ct.capture_image(cam)
                x, y = ct.locate(img)
                if x is not None:
                    print(x,y)
                    tmeas1 = api.getPos(0,module)
                    tmeas2 = api.getPos(1,module)
                    # diff
                    xr = x - self.xy[0]
                    yr = y - self.xy[1]
                    tr1 = tmeas1 - self.t[0]
                    tr2 = tmeas2 - self.t[1]
                    
                    self.data[self.i,:] = np.array([tr1, tr2, xr, yr])
                    # self.data[self.i,:] = np.array([tr1, tr2, tmeas1, tmeas2, xr, yr]) # for angle
                    self.i += 1
                    # print(self.i,xr,yr)

                    # set new pos
                    if self.i != num_datapoints:
                        # api.setPos(thetas[0,self.i], thetas[1,self.i], module)
                        api.setPos(*get_r_angle(), module)
                        self.time_of_move = datetime.datetime.now()
                        self.xy = [None,None]
                        self.t  = [None,None]
                else:
                    print("Obj not found")
                    return False

test = TestClass(num_datapoints)
test.go()

print('Terminating')
api.terminate()


# TODO SAVE .csv file with robot pos data and target location x,y
