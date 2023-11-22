import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from scipy import stats

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