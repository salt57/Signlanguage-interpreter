import tensorflow as tf
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import os
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2
import mediapipe as mp


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
#Extracting the keypoint and landmark information for each action from its storage location and splitting all the data into respective training and testing datasets
actions = np.array(['yes', 'no', 'i love you', 'happy'])
DATA_PATH = os.path.join('MP_Data_new')
framesPerVid = 60
label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []

for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frameNum in range(framesPerVid):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frameNum)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

import keras.backend as K
import keras
@keras.saving.register_keras_serializable(package="my_package", name="custom_fn")
def custom_activation(x):
    sigmoid_part = K.sigmoid(x)
    tanh_part = K.tanh(x)
    output = sigmoid_part * tanh_part

    return output

model = tf.keras.models.load_model('Assets/test.keras', custom_objects={'custom_activation':custom_activation})
print(model.summary())

ytrue = np.argmax(y_test, axis=1).tolist()
ypred = np.argmax(model.predict(X_test), axis=1).tolist()

print(multilabel_confusion_matrix(ytrue, ypred))
print(accuracy_score(ytrue, ypred))

#Applying the model in real time to each frame of data recieved from the user in order to detect the signs being displayed
# model = tf.keras.models.load_model('signlangnew.h5')
actions = np.array(['yes', 'no', 'i love you', 'happy', 'sad'])
sequence = []
full = ''
sentence = ''
prev_sen = ''
predictions = []
threshold = 0.8

cap = cv2.VideoCapture(0)
with mpHolistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        ret, frame = cap.read()
        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(image, results)
        
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        #print(len(sequence))
        sequence = sequence[-60:]
        #print(np.expand_dims(sequence, axis=0).shape)
        
        if len(sequence) == 60:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            #print(res)
            predictions.append(np.argmax(res))
            # print(predictions)
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold:                     
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence:
                            sentence = actions[np.argmax(res)]
                    else:
                        sentence = actions[np.argmax(res)]

            # if res[prediction] > threshold:                     
            #     if len(sentence) > 0: 
            #         if actions[np.argmax(res)] != sentence:
            #             sentence = actions[np.argmax(res)]
            #     else:
            #         sentence = actions[np.argmax(res)]
            # image = prob_viz(actions[np.argmax(res)], image, colors)
            
        cv2.rectangle(image, (0,0), (640, 100), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(full), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('OpenCV Feed', image)
        import os
        from gtts import gTTS

        if sentence != prev_sen:
            full += sentence + " "
            # tts = gTTS(sentence)
            # tts.save("Assets/output.mp3")
            os.system(f"say {sentence}")
            # os.system("mpg321 Assets/output.mp3")  # On Linux
            # os.system("afplay Assets/output.mp3")  # On macOS
            # os.system("start Assets/output.mp3")
            prev_sen = sentence

        if cv2.waitKey(10) & 0xFF == ord('q'):
            if full:
                sec = gTTS(full)
                sec.save("Assets/full.mp3")
            break
    cap.release()
    cv2.destroyAllWindows()