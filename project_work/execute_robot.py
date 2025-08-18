import numpy as np
import torch
import utils.torch_model as torch_model
import cv2
import utils.camera_tools as ct
from FableAPI.fable_init import api
from datetime import datetime

test_second = True

cam = ct.prepare_camera()
print(cam.isOpened())  # False
i = 0

data = np.loadtxt("data.csv", delimiter=",")
end_pos = data[:,2:].T
x = torch.from_numpy(end_pos.T).float()
data_mean = x.mean(axis=0)

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

# TODO Load the trained model
model1 = torch_model.MLPNet(2, 24, 2)
model1.load_state_dict(torch.load('650_trained_model.pth'))
if test_second:
    model2 = torch_model.MLPNet(2, 24, 2)
    model2.load_state_dict(torch.load('active_model_9.pth'))

# dummy class for targets
class CoordinateStore:
    def __init__(self):
        self.point = None
        self.new = False

    def select_point(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(frame,(x,y),3,(255,0,0),-1)
            self.point = [x,y]
            self.new = True


# Instantiate class
coordinateStore1 = CoordinateStore()

# as alternative you can set prior targets

cv2.namedWindow("test")
cv2.setMouseCallback('test', coordinateStore1.select_point)
set_time = 0

while True:

    frame = ct.capture_image(cam)

    x, y = ct.locate(frame)

    cv2.imshow("test", frame)
    k = cv2.waitKey(100)
    if k == 27:
        break

    # print(coordinateStore1.point)
    # print(coordinateStore1.new)
    # get the prediction
    if coordinateStore1.new:
        inp = torch.tensor([coordinateStore1.point]).float()
        inp -= data_mean
        # shows the non-active learning model prediction
        with torch.no_grad():
            outp = model1(inp)
            t = outp.numpy()[0]  
            print(t)
        api.setPos(t[0], t[1], module)
        if not test_second:
            continue
        set_time = datetime.now()
        # shows the active learning model prediction
        while (datetime.now() - set_time).total_seconds() < 3.0:
            frame = ct.capture_image(cam)
            ct.locate(frame)
            cv2.imshow("test", frame)
            k = cv2.waitKey(100)
        with torch.no_grad():
            outp = model2(inp)
            t = outp.numpy()[0]
            print(t)
        api.setPos(t[0], t[1], module)
        coordinateStore1.new = False

print('Terminating')
api.terminate()

