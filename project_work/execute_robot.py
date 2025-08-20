import numpy as np
import torch
import cv2
import utils.torch_model as torch_model
import utils.camera_tools as ct
import hand_tracker as ht
from FableAPI.fable_init import api
from datetime import datetime


enable_point_bounding = True

cam = ct.prepare_camera()
print(cam.isOpened())  # False
i = 0

# data = np.loadtxt("data.csv", delimiter=",")
# end_pos = data[:,2:].T
# x = torch.from_numpy(end_pos.T).float()
# data_mean = x.mean(axis=0)
data_mean = torch.tensor([284.9700, 318.9700]).float()

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

vlen = lambda xy: (xy[0]**2 + xy[1]**2)**(1/2)
def point_bounding(xy):
    # top    [322, 255]
    # left   [153, 426]
    # center [322, 426]
    # radius [169, 171] ~ 170
    pc = torch.tensor([322, 426]).float()
    xyc = xy - pc
    if vlen(xyc) > 170:
        xy = xyc/vlen(xyc)*170 + pc
    return xy

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
model = torch_model.MLPNet(2, 24, 2)
model.load_state_dict(torch.load('280_trained_model.pth'))

# Instantiate tracker
hand_landmarker = ht.landmarker_and_result(1)

# as alternative you can set prior targets

cv2.namedWindow("hand tracker")
frame = ct.capture_image(cam)
H,W = frame.shape[:2]
while True:
    frame = ct.capture_image(cam)
    # x, y = ct.locate(frame)

    cv2.imshow("hand tracker", frame)
    k = cv2.waitKey(20)
    if k == 27:
        break
    hand_landmarker.detect_async(frame)
    xc,yc,*_ = hand_landmarker.get_points()[0]
    
    # get the prediction
    if xc != None and yc != None:
        inp = torch.tensor([xc*W,yc*H]).float()
        if enable_point_bounding:
            inp = point_bounding(inp)
        inp -= data_mean
        # MLP
        with torch.no_grad():
            outp = model(inp)
            t = outp.numpy()  
            print(t)
        api.setPos(abs(t[0]), t[1], module)

print('Terminating')
api.terminate()

