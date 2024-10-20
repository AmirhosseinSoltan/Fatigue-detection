#!/usr/bin/env python3
import cv2
import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import threading
import matplotlib.pyplot as plt


BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Function to add text on the frame
def add_text(frame, text, position=(50, 50), font=cv2.FONT_HERSHEY_SIMPLEX, 
             font_scale=1, font_color=(105, 225, 0), thickness=2):
    cv2.putText(frame, text, position, font, font_scale, font_color, thickness)


def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image


# Global variable to store the latest annotated image
latest_frame = None
frame_num = 0
frame_ready : bool = False
lock = threading.Lock() 

''' Lock to ensure safe access the shared last frame variable
In this case, we are running multiple threads. The main body of the program
is running continuously, trying to read the input frames and dispaly them.
However, the callback function which detects landmarks is called "asynchronously".
When two threads try to access/modify the same variable, it causes the race condition,
incocsistancy and erros... 
So we lock the global-shared variable ('latest_frame') when trying to access it (with lock: ... )
Python automatically will realase the lock after the operation is complete.
It ensures that only one thread (either the callback or the main loop) is 
accessing or modifying latest_frame at a time.

I also considered the frame_ready flag to make the process cosistant.
'''

def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    
    global latest_frame, frame_ready, lock, frame_num

    image_data = output_image.numpy_view()[:,:,:3]

    # print(result.face_landmarks[0])

    annotated_image = draw_landmarks_on_image(image_data, result)
    image_result = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    
    # Safely update the global frame with the new image
    with lock:
        latest_frame = image_result
        frame_ready = True
    
    print(f"Updated latest_frame with frame num = {frame_num} and shape{latest_frame.shape}")  # Debug print

base_options = python.BaseOptions(model_asset_path='../model/face_landmarker.task')
                                  # delegate_options=mp.tasks.BaseOptions.Cpu(use_xnnpack=False))

options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1,
                                       running_mode = vision.RunningMode.LIVE_STREAM,
                                       result_callback= print_result)

''' 
detect_async will provide face landmark detection results to the 'result_callback'
result_callback 
    The `result_callback` provides:
      - The face landmarks detection results.
      - The input image that the face landmarker runs on.
      - The input timestamp in milliseconds.

'''

if __name__ == '__main__':
  with FaceLandmarker.create_from_options(options) as landmarker:
      cap = cv2.VideoCapture(0)
      while True:
          succes, img = cap.read()
          if not succes:
              print("Ignoring empty camera frame.")
              continue
          
          timestamp_ms = int(time.time() * 1000)
          
          frame_num += 1
          imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
          mp_RGB_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
          landmarker.detect_async(mp_RGB_frame, timestamp_ms)

            # Safely retrieve the latest annotated frame
          with lock:
              if frame_ready and latest_frame is not None:
                  
                  add_text(latest_frame,text=f'Frame num ={frame_num}, timestamp = {timestamp_ms}')
                  cv2.imshow("Live-Stream Face landmark detector", latest_frame)
                  frame_ready = False  # Swaping the flag


          # cv2.waitKey(10)

          if cv2.waitKey(1) & 0xFF == 27 :
            break

      cap.release()
      cv2.destroyAllWindows()
