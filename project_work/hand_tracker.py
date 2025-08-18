import cv2
import numpy as np
import time

import mediapipe as mp

class landmarker_and_result():
    def __init__(self, n_hands = 2):
        self.result = mp.tasks.vision.HandLandmarkerResult
        self.landmarker = mp.tasks.vision.HandLandmarker
        self.createLandmarker(n_hands)
    
    def createLandmarker(self, n_hands):
        # callback function
        def update_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            self.result = result

        # HandLandmarkerOptions (details here: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python#live-stream)
        options = mp.tasks.vision.HandLandmarkerOptions( 
            base_options = mp.tasks.BaseOptions(model_asset_path="./project_work/hand_landmarker.task"), # path to model
            # base_options = mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"), # path to model
            running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM, # running on a live stream
            num_hands = n_hands, # track both hands
            min_hand_detection_confidence = 0.1, # lower than value to get predictions more often
            min_hand_presence_confidence = 0.2, # lower than value to get predictions more often
            min_tracking_confidence = 0.2, # lower than value to get predictions more often
            result_callback=update_result)
        
        # initialize landmarker
        self.landmarker = self.landmarker.create_from_options(options)
    
    def detect_async(self, frame):
        # convert np frame to mp image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        # detect landmarks
        self.landmarker.detect_async(image = mp_image, timestamp_ms = int(time.time() * 1000))

    def close(self):
        # close landmarker
        self.landmarker.close()
        
    def get_points(self):
        try:
            hand_landmarks_list = self.result.hand_landmarks
            if hand_landmarks_list == []:
                return []
            centers = []
            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]
                x_coordinates = [landmark.x for landmark in hand_landmarks]
                y_coordinates = [landmark.y for landmark in hand_landmarks]
                x_min,x_max = min(x_coordinates),max(x_coordinates)
                y_min,y_max = min(y_coordinates),max(y_coordinates)
                x_c = (x_max+x_min)/2
                y_c = (y_max+y_min)/2
                centers.append([x_c,y_c,x_min,x_max,y_min,y_max])
            print("GOT CENTERS",len(centers))
            return centers
        except:
            return []

    def draw_result_on_image(self, rgb_image):
        """Courtesy of https://github.com/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb"""
        try:
            if self.result.hand_landmarks == []:
                return rgb_image
            hand_landmarks_list = self.result.hand_landmarks
            annotated_image = np.copy(rgb_image)
            H,W = annotated_image.shape[:2]
            # print(W,H)

            # Loop through the detected hands to visualize.
            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]
                x_coordinates = [landmark.x for landmark in hand_landmarks]
                y_coordinates = [landmark.y for landmark in hand_landmarks]
                cv2.rectangle(annotated_image, 
                            (int(min(x_coordinates)*W),int(min(y_coordinates)*H)), 
                            (int(max(x_coordinates)*W),int(max(y_coordinates)*H)),
                            (0,0,255),2)
                # print(min(x_coordinates),max(x_coordinates))
                
                # # Draw the hand landmarks.
                # hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                # hand_landmarks_proto.landmark.extend([
                #    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
                # mp.solutions.drawing_utils.draw_landmarks(
                #    annotated_image,
                #    hand_landmarks_proto,
                #    mp.solutions.hands.HAND_CONNECTIONS,
                #    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                #    mp.solutions.drawing_styles.get_default_hand_connections_style())
            return annotated_image
        except:
            return rgb_image



def main(hand_landmarker):
    # access webcam
    cap = cv2.VideoCapture(0)
    while True:
        # pull frame
        ret, frame = cap.read()
        # mirror frame
        frame = cv2.flip(frame, 1)
        # update landmarker results
        hand_landmarker.detect_async(frame)
        # draw landmarks on frame
        frame = hand_landmarker.draw_result_on_image(frame)
        # display frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(20) == ord('q'):
            break

    # release everything
    cap.release()
    cv2.destroyAllWindows()
    hand_landmarker.close()

if __name__ == "__main__":
    # create landmarker
    hand_landmarker = landmarker_and_result(n_hands=1)
    main(hand_landmarker)
