import mediapipe as mp
import cv2
import numpy as np
import os

#Defining mediapipe holistic model and model to draw landmarks

mpHolistic = mp.solutions.holistic 
mpDraw = mp.solutions.drawing_utils
actions = np.array(['yes', 'no', 'i love you', 'happy'])


#Function to take in a frame as input, extract face, hand and pose landmarks and return it to the user

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                  
    results = model.process(image)                 
    image.flags.writeable = True                    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

#Function that takes image and its landmarks as input and uses the mediapipe drawing model to draw landmarks

def draw_landmarks(image, results):
    mpDraw.draw_landmarks(image, results.pose_landmarks, mpHolistic.POSE_CONNECTIONS,
                             mpDraw.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mpDraw.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mpDraw.draw_landmarks(image, results.left_hand_landmarks, mpHolistic.HAND_CONNECTIONS, 
                             mpDraw.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mpDraw.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mpDraw.draw_landmarks(image, results.right_hand_landmarks, mpHolistic.HAND_CONNECTIONS, 
                             mpDraw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mpDraw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

#Function to extract the keypoints from the face, hand and pose landmarks and return the values in one array

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

#Function to visualize and render the probability of a particular action live on the OpenCV display interface

colors = (117,245,16)

def prob_viz(action, input_frame, colors):
    output_frame = input_frame.copy()
    cv2.rectangle(output_frame, (0,60), (90), colors, -1)
    cv2.putText(output_frame, action, (0, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)        
    return output_frame

#Setting up storage spaces for storing each frame of data for every video for each action we are training the model on

try: 
    os.mkdir('MP_Data_new')
except:
    pass

DATA_PATH = os.path.join('MP_Data_new') 

# Actions that we try to detect
actions = np.array(['yes', 'no', 'i love you', 'happy'])

for action in actions:
    try: 
        os.makedirs(os.path.join(DATA_PATH, action))
    except:
        pass

numVids = 20

framesPerVid = 60

for action in actions: 
    for sequence in range(numVids):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

#Collection of data on the keypoints and landmark positions for each sign language action in order to train the model on it
numVids = 20
DATA_PATH = os.path.join('MP_Data_new') 
framesPerVid = 60
i = 0
cap = cv2.VideoCapture(0)
action = actions[i]

with mpHolistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for sequence in range(numVids):
        for frameNum in range(framesPerVid):
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results)

            if frameNum == 0: 
                cv2.putText(image, 'STARTING COLLECTION', (150,200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence + 1), (130,250), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(1000)
            else: 
                cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence + 1), (15,12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)

            keypoints = extract_keypoints(results)
            npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frameNum))
            np.save(npy_path, keypoints)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break              
cap.release()
cv2.destroyAllWindows() 
