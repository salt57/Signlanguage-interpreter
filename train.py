import os
import numpy as np
import keras
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from matplotlib import pyplot as plt

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

#Building the LSTM and Dense layers and training the model on the training dataset, as well as logging its live accuracy and loss percentage with each epoch

import keras.backend as K
@keras.saving.register_keras_serializable(package="my_package", name="custom_fn")
def custom_activation(x):
    sigmoid_part = K.sigmoid(x)
    tanh_part = K.tanh(x)
    output = sigmoid_part * tanh_part

    return output

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='sigmoid', input_shape=(60,258)))
model.add(LSTM(128, return_sequences=True, activation='sigmoid'))
model.add(LSTM(64, return_sequences=False, activation='sigmoid'))
# model.add(LSTM(64, return_sequences=True, activation=custom_activation, input_shape=(60,258)))
# model.add(LSTM(128, return_sequences=True, activation=custom_activation))
# model.add(LSTM(64, return_sequences=False, activation=custom_activation))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model2 = Sequential()
# model.add(LSTM(64, return_sequences=True, activation='sigmoid', input_shape=(60,258)))
# model.add(LSTM(128, return_sequences=True, activation='sigmoid'))
# model.add(LSTM(64, return_sequences=False, activation='sigmoid'))
model2.add(LSTM(64, return_sequences=True, activation=custom_activation, input_shape=(60,258)))
model2.add(LSTM(128, return_sequences=True, activation=custom_activation))
model2.add(LSTM(64, return_sequences=False, activation=custom_activation))
model2.add(Dense(64, activation='relu'))
model2.add(Dense(32, activation='relu'))
model2.add(Dense(actions.shape[0], activation='softmax'))
model2.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# %tensorboard --logdir logs
history = model.fit(X_train, y_train, epochs=75, callbacks=[tb_callback])
history2 = model2.fit(X_train, y_train, epochs=75, callbacks=[tb_callback])
model.summary()

model2.save('Assets/test.keras')

ytrue = np.argmax(y_test, axis=1).tolist()
ypred = np.argmax(model.predict(X_test), axis=1).tolist()

print(multilabel_confusion_matrix(ytrue, ypred))
print(accuracy_score(ytrue, ypred))

from matplotlib import pyplot as plt

plt.plot(history.history['categorical_accuracy'], color='blue')
plt.plot(history2.history['categorical_accuracy'], color='red')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['sigmoid', 'custom'], loc='upper left')
plt.show()

plt.plot(history.history['loss'], color='blue')
plt.plot(history2.history['loss'], color='red')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['sigmoid', 'custom'], loc='upper left')
plt.show()

model.save('Assets/signlangnew.h5')