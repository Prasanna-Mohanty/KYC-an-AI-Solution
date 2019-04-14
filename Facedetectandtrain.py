# Import OpenCV2 for image processing
import cv2
import os

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Start capturing video
vid_cam = cv2.VideoCapture(0)

# Detect object in video stream using Haarcascade Frontal Face
#face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#face_detector = cv2.CascadeClassifier('C:/Users/Prasanna Mohanty/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
#face_detector = cv2.CascadeClassifier('C:/Users/e3003895/AppData/Local/Continuum/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('C:/Users/e3003895/AppData/Local/Continuum/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/e3003895/AppData/Local/Continuum/anaconda3/Lib/site-packages/cv2/data/haarcascade_eye.xml')


# For each person, one face id
#face_id = 1
#face_id = 4

# For each person, enter one numeric face id
face_id = input('\n enter Customer id end press <return> ==>  ')

# Initialize sample face image
count = 0

assure_path_exists("dataset/")

# Start looping
while(True):

    # Capture video frame
    _, image_frame = vid_cam.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    # Detect frames of different sizes, list of faces rectangles
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print('Number of faces detected: ' + str(len(faces)))
    # Loops for each faces
    for (x,y,w,h) in faces:

        # Crop the image frame into rectangle
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
        #pm
        roi_gray = gray[y:y+h, x:x+w]

        roi_color = image_frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex,ey,ew,eh) in eyes:

            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        #pm
        # Increment sample face image
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/Customer." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        # Display the video frame, with bounded rectangle on the person's face
        #cv2.imshow('frame', image_frame)
        cv2.imshow('image_frame', image_frame) 

    # To stop taking video, press 'q' for at least 100ms
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    # If image taken reach 100, stop taking video
    elif count>99:
        break
######Starting the Training Module
# Import OpenCV2 for image processing
# Import os for file path
import cv2, os

# Import numpy for matrix calculation
import numpy as np

# Import Python Image Library (PIL)
from PIL import Image

import os

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Using prebuilt frontal face training model, for face detection
#detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
#detector = cv2.CascadeClassifier('C:/Users/Prasanna Mohanty/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
detector = cv2.CascadeClassifier('C:/Users/e3003895/AppData/Local/Continuum/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

# Create method to get the images and label data
def getImagesAndLabels(path):

    # Get all file path
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 
    
    # Initialize empty face sample
    faceSamples=[]
    
    # Initialize empty id
    ids = []

    # Loop all the file path
    for imagePath in imagePaths:

        # Get the image and convert it to grayscale
        PIL_img = Image.open(imagePath).convert('L')

        # PIL image to numpy array
        img_numpy = np.array(PIL_img,'uint8')

        # Get the image id
        id = int(os.path.split(imagePath)[-1].split(".")[1])

        # Get the face from the training images
        faces = detector.detectMultiScale(img_numpy)

        # Loop for each face, append to their respective ID
        for (x,y,w,h) in faces:

            # Add the image to face samples
            faceSamples.append(img_numpy[y:y+h,x:x+w])

            # Add the ID to IDs
            ids.append(id)

    # Pass the face array and IDs array
    return faceSamples,ids

# Get the faces and IDs
faces,ids = getImagesAndLabels('dataset')

# Train the model using the faces and IDs
recognizer.train(faces, np.array(ids))

# Save the model into trainer.yml
assure_path_exists('trainer/')
recognizer.save('trainer/trainer.yml')
######End of the Training Module
# Stop video
vid_cam.release()

# Close all started windows
cv2.destroyAllWindows()