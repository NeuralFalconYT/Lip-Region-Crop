video_path= "./video.mp4"
# video_path="0"
# fixed_width,fixed_height = 640,480
# fixed_width,fixed_height = 1280,720
fixed_width,fixed_height =  1920,1080
# fixed_width,fixed_height =  1080,1920
import mediapipe as mp
import cv2
import time
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

model_path = './face_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face landmarker instance with the video mode:
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)

landmarker=FaceLandmarker.create_from_options(options)
 




def draw_landmarks_on_image(frame, detection_result):
# Convert the BGR image to RGB before processing.
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
    # Convert the RGB image back to BGR.
    annotated_image=cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    return annotated_image

def normalized_to_pixel_coordinates(normalized_coordinates, image_width, image_height):
    normalized_x, normalized_y,normalized_z = normalized_coordinates
    return normalized_x, normalized_y

def get_landmarks(image_width,image_height,face_landmarker_result):
    all_landmarks = []

    for face_landmarks in face_landmarker_result.face_landmarks:
        landmarks_coordinates={}
        i=0
        for landmark in face_landmarks:
            pixel_x = int(landmark.x * image_width)
            pixel_y = int(landmark.y * image_height)
            pixel_z = int(landmark.z * image_width)
            landmarks_coordinates[i]=(pixel_x, pixel_y, pixel_z)
            i+=1
        all_landmarks.append(landmarks_coordinates)
    return all_landmarks

old_x1,old_y1,old_x2,old_y2=0,0,0,0
x1,y1,x2,y2=0,0,0,0
old_lip_top_x1,old_lip_top_y1,old_lip_down_x2, old_lip_down_y2 =0,0,0,0
lip_top_x1,lip_top_y1,lip_down_x2, lip_down_y2=0,0,0,0
old_frame=None
def process_image(frame,face_landmark):
    global old_x1,old_y1,old_x2,old_y2
    global x1,y1,x2,y2
    global old_lip_top_x1,old_lip_top_y1,old_lip_down_x2, old_lip_down_y2 
    global lip_top_x1,lip_top_y1,lip_down_x2, lip_down_y2
    global old_frame
    # x1,y1,x2,y2,lip_top_x1,lip_top_y1,lip_down_x2, lip_down_y2
    # Update the section where you access landmarks
    lip_image=frame.copy()
    box_image=frame.copy()
    if face_landmark and len(face_landmark[0]) > 93:
        x1, y1, _ = face_landmark[0][93]
        old_x1, old_y1 = x1, y1
    else:
        x1, y1 = old_x1, old_y1

    if face_landmark and len(face_landmark[0]) > 365:
        x2, y2, _ = face_landmark[0][365]
        old_x2, old_y2 = x2, y2
    else:
        x2, y2 = old_x2, old_y2

    # Similarly, update for lip_top landmarks
    if face_landmark and len(face_landmark[0]) > 0:
        lip_top_x1, lip_top_y1, _ = face_landmark[0][0]
        old_lip_top_x1, old_lip_top_y1 = lip_top_x1, lip_top_y1
    else:
        lip_top_x1, lip_top_y1 = old_lip_top_x1, old_lip_top_y1

    if face_landmark and len(face_landmark[0]) > 17:
        lip_down_x2,  lip_down_y2, _ = face_landmark[0][17]
        old_lip_down_x2, old_lip_down_y2 = lip_down_x2, lip_down_y2
    else:
        lip_down_x2, lip_down_y2= old_lip_down_x2, old_lip_down_y2


    if face_landmark and len(face_landmark[0]) > 287 and len(face_landmark[0]) > 57:
        left_lip_x,left_lip_y,_= face_landmark[0][287]
        right_lip_x,right_lip_y,_=face_landmark[0][57]
        if left_lip_y > right_lip_y:
            A = (right_lip_x, left_lip_y)
        else:
            A = (left_lip_x, right_lip_y)
        # Draw the points on the image
        lip_image = cv2.circle(lip_image, A, 5, (0, 0, 255), -1)
        lip_image = cv2.circle(lip_image, (left_lip_x,left_lip_y), 5, (0, 0, 255), -1)
        lip_image = cv2.circle(lip_image, (right_lip_x,right_lip_y), 5, (0, 0, 255), -1)
        lip_image = cv2.line(lip_image, (left_lip_x,left_lip_y), (right_lip_x,right_lip_y), (0, 255, 0), 2)
        lip_image = cv2.line(lip_image, (left_lip_x,left_lip_y), A, (0, 255, 0), 2)
        lip_image = cv2.line(lip_image, (right_lip_x,right_lip_y), A, (0, 255, 0), 2)
  # Integer 1 indicates that ima
  
        delta_x = right_lip_x - left_lip_x
        delta_y = right_lip_y - left_lip_y
        angle=np.arctan(delta_y/delta_x)
        angle = (angle * 180) / np.pi
        # Width and height of the image
        h, w = frame.shape[:2]
        # Calculating a center point of the image
        # Integer division "//"" ensures that we receive whole numbers
        center = (w // 2, h // 2)
        # Defining a matrix M and calling
        # cv2.getRotationMatrix2D method
        M = cv2.getRotationMatrix2D(center, (angle), 1.0)
        # Applying the rotation to our image using the
        # cv2.warpAffine method
        rotated = cv2.warpAffine(frame, M, (w, h))
        frame = rotated
        # update points
        rotated_points = np.dot(M, np.array([[x1], [y1], [1]]))
        x1, y1 = int(rotated_points[0]), int(rotated_points[1])
        
        rotated_points = np.dot(M, np.array([[x2], [y2], [1]]))
        x2, y2 = int(rotated_points[0]), int(rotated_points[1])
        
        rotated_points = np.dot(M, np.array([[lip_top_x1], [lip_top_y1], [1]]))
        lip_top_x1, lip_top_y1 = int(rotated_points[0]), int(rotated_points[1])
        
        rotated_points = np.dot(M, np.array([[lip_down_x2], [lip_down_y2], [1]]))
        lip_down_x2, lip_down_y2 = int(rotated_points[0]), int(rotated_points[1])
        box_image=frame.copy()
        box_image = cv2.circle(box_image, (lip_top_x1, lip_top_y1), 5, (0, 0, 255), -1)
        box_image = cv2.circle(box_image, (lip_down_x2, lip_down_y2), 5, (0, 0, 255), -1)
    try:    
        Y = y2 - y1
        # Calculate new_y1 and new_y2 based on lip points
        new_y1 = y1 + (lip_top_y1 - y1)-12
        new_y2 = y2 + (lip_down_y2 - y2)+9
        box_image = cv2.rectangle(box_image, (x1, new_y1), (x2, new_y2), (0, 255, 0), 2)
        box_image = cv2.circle(box_image, (x1, new_y1), 5, (0, 0, 255), -1)
        box_image = cv2.circle(box_image, (x2, new_y2), 5, (0, 0, 255), -1)
        roi = frame[new_y1:new_y2, x1+25:x2]
        roi = cv2.resize(roi, (roi.shape[1]*2, roi.shape[0]*2))
        #i want to give a fixed size to the roi
        # so first create a black image with the fixed size like 640x480
        # then paste the roi in the center of the black image but keep the aspect ratio of
        # the roi
        
        # cv2.imshow('crop', roi)
        old_frame=roi
        return roi,True,frame,lip_image,box_image
    except:
        if old_frame is not None:
            return old_frame,False,frame,lip_image,box_image
        else:
            return frame,False,frame,lip_image,box_image


import cv2
import numpy as np

# fixed_width,fixed_height = 640,480
# fixed_width,fixed_height = 1280,720
# fixed_width,fixed_height =  1920,1080
def resize_and_paste_roi(roi):
    # Desired fixed size for the canvas
    global fixed_width,fixed_height 



    # Calculate aspect ratio of roi
    roi_height, roi_width = roi.shape[:2]
    aspect_ratio = roi_width / roi_height

    # Calculate new dimensions to fit within fixed size
    if aspect_ratio > 1:  # Landscape orientation
        new_width = fixed_width
        new_height = int(fixed_width / aspect_ratio)
    else:  # Portrait orientation or square
        new_height = fixed_height
        new_width = int(fixed_height * aspect_ratio)

    # Resize roi while maintaining aspect ratio
    roi_resized = cv2.resize(roi, (new_width, new_height))

    # If resized roi is larger than the canvas, resize again to fit within canvas
    if roi_resized.shape[0] > fixed_height or roi_resized.shape[1] > fixed_width:
        roi_resized = cv2.resize(roi_resized, (fixed_width, fixed_height))

    # Create black canvas
    canvas = np.zeros((fixed_height, fixed_width, 3), dtype=np.uint8)

    # Calculate position to paste roi in the center of the canvas
    x_offset = (fixed_width - roi_resized.shape[1]) // 2
    y_offset = (fixed_height - roi_resized.shape[0]) // 2

    # Paste resized roi onto the canvas
    canvas[y_offset:y_offset+roi_resized.shape[0], x_offset:x_offset+roi_resized.shape[1]] = roi_resized

    return canvas











# Initialize VideoCapture object
if video_path=="0":
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:  # If fps is not properly detected, set a default value
    fps = 30.0

#video save
output_filename = 'temp.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
frame_width =fixed_width
frame_height = fixed_height

# frame_width, frame_height = 1278,480
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

# Record the start time
start_time = time.time()

# Initialize previous timestamp
previous_timestamp_ms = 0

# Loop through each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    # frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    if not ret:
        break
    image_width, image_height=frame.shape[1],frame.shape[0]    
    # Calculate the elapsed time in milliseconds
    elapsed_time = time.time() - start_time
    frame_timestamp_ms = int(elapsed_time * 1000)  # Convert to milliseconds

    # Ensure the timestamp is monotonically increasing
    if frame_timestamp_ms <= previous_timestamp_ms:
        frame_timestamp_ms = previous_timestamp_ms + 1
    
    previous_timestamp_ms = frame_timestamp_ms

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert the frame to a MediaPipe Image object
    mp_image_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
    # Detect faces in the frame using MediaPipe
    face_landmarker_result = landmarker.detect_for_video(mp_image_frame, frame_timestamp_ms)
    annotated_image=draw_landmarks_on_image(frame, face_landmarker_result)
    face_landmark=get_landmarks(image_width,image_height,face_landmarker_result)
    roi,check_lip,align_frame,lip_image,box_image=process_image(frame,face_landmark)
    #resize roi make it larger but keep the aspect ratio
    roi = cv2.resize(roi, (roi.shape[1]*1, roi.shape[0]*1))
    # cv2.imshow('crop', roi)
    crop_lip = resize_and_paste_roi(roi)
    # if check_lip:
    #     out.write(crop_lip)
    out.write(crop_lip)    
    format_size=(426,240)    
    annotated_image=cv2.resize(annotated_image,format_size)
    lip_image=cv2.resize(lip_image, format_size)
    box_image=cv2.resize(box_image, format_size)
    crop_lip = cv2.resize(crop_lip, format_size)
    align_frame = cv2.resize(align_frame, format_size)
    frame = cv2.resize(frame, format_size)
    #put text on all the images in top left corner
    frame=cv2.putText(frame, "1. Original", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    annotated_image=cv2.putText(annotated_image, "2. Face Landmark", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    lip_image=cv2.putText(lip_image, "3. Lip Points", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    align_frame=cv2.putText(align_frame, "4. Face Alignment", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    box_image=cv2.putText(box_image, "5. Lip Box", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    crop_lip=cv2.putText(crop_lip, "6. Cropped Lip", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    first_frame = cv2.hconcat([frame,annotated_image,lip_image])
    second_frame = cv2.hconcat([align_frame,box_image,crop_lip])
    third_frame = cv2.vconcat([first_frame,second_frame])
    third_frame=third_frame.astype(np.uint8)
    # print(third_frame.shape)
    # out.write(third_frame)
    cv2.imshow('final_frame', third_frame)
    
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all OpenCV windows
cap.release()
out.release()
cv2.destroyAllWindows()
