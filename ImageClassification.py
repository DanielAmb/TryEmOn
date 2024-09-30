import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

from tensorflow.python.keras import layers

from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(type(x_train))
print(type(y_train))
print(type(x_test))
print(type(y_test))

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

# index = 0
# x_train[index]
# img = plt.imshow(x_train[index])
# print('The image label is:', y_train[index])

classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#print('The image class is:', classification[y_train[index]])




y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)
#print(y_train_one_hot)

#print('The one hot label is:', y_train_one_hot[index])

x_train = x_train / 255
x_test = x_test / 255
#x_train[index]

model = Sequential()

model.add(Conv2D(32, (5,5), activation='relu', input_shape=(32,32,3)))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(32, (5,5), activation='relu'))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())

model.add(Dense(1000, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(500, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(250, activation='relu'))

model.add(Dense(10, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(x_train, y_train_one_hot, batch_size = 256, epochs = 10, validation_split = 0.2)

model.evaluate(x_test, y_test_one_hot)[1]

# plt.plot(hist.history['accuracy'])
# plt.plot(hist.history['val_accuracy'])
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Val'], loc='upper left')
# plt.show()

# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Val'], loc='upper right')
# plt.show()

#from google.colab import files
#uploaded = files.upload()

new_image = plt.imread('Cat.jpg')
img = plt.imshow(new_image)

from skimage.transform import resize
resized_image = resize(new_image, (32,32,3))
img = plt.imshow(resized_image)

predictions = model.predict(np.array([resized_image]))

list_index = [0,1,2,3,4,5,6,7,8,9]
x = predictions
for i in range(10):
    for j in range(10):
        if x[0][list_index[i]] > x[0][list_index[j]]:
            temp = list_index[i]
            list_index[i] = list_index[j]
            list_index[j] = temp

print(list_index)

for i in range(5):
    print(classification[list_index[i]], ':', round(predictions[0][list_index[i]] * 100, 2), '%')



