import os
import numpy as np
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

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='sigmoid', input_shape=(60,258)))
model.add(LSTM(128, return_sequences=True, activation='sigmoid'))
model.add(LSTM(64, return_sequences=False, activation='sigmoid'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# %tensorboard --logdir logs
history = model.fit(X_train, y_train, epochs=500, callbacks=[tb_callback])
model.summary()

model.save('Assets/test.h5')

ytrue = np.argmax(y_test, axis=1).tolist()
ypred = np.argmax(model.predict(X_test), axis=1).tolist()

print(multilabel_confusion_matrix(ytrue, ypred))
print(accuracy_score(ytrue, ypred))

from matplotlib import pyplot as plt

plt.plot(history.history['categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

model.save('Assets/signlangnew.h5')