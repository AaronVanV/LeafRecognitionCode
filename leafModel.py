#!/N/u/
'''
Model using batch Normalization as regularization method and relu as Activation function
'''

from keras.preprocessing.image import ImageDataGenerator, array_to_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import CSVLogger
from tensorflow.keras.layers import BatchNormalization
from matplotlib import pyplot as plt
import pandas as pd



#Images dimensions
img_width, img_height = 324, 324

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 900
nb_validation_samples = 225
epochs = 100
batch_size = 45


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#layer 1
model = Sequential()
model.add(Conv2D(32, (5, 5), strides=(2,2), use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', input_shape = input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))

#layer 2
model.add(Conv2D(32, (5, 5), strides=(2,2), use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))

#layer 3
model.add(Conv2D(64, (5, 5), strides=(1,1), use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
model.add(Activation('relu'))

#layer 4
model.add(Conv2D(64, (5, 5), strides=(1,1), use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2),strides=(1, 1)))

#flatten
model.add(Flatten(input_shape=(11, 11)))

#dense layer Classifier
model.add(Dense(64))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(32))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(15))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))
model.add(Activation('softmax'))

#compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#data augmentation for training
train_datagen = ImageDataGenerator()

#data augmentation for testing:
valid_datagen = ImageDataGenerator()

#take path to dir and generate batches of augmented data
train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode='categorical')

validation_generator = valid_datagen.flow_from_directory(validation_data_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode='categorical')

csv_logger = CSVLogger('leaf_v1-1.csv');

#model training
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[csv_logger])




model.save("model", save_format="h5")


df = pd.read_csv("leaf_v1-1.csv")

plt.xlabel("epoch")
plt.ylabel("amount")
plt.ylim(0, 1.2)
plt.plot(df["epoch"],df["accuracy"],label='accuracy')
plt.plot(df["epoch"],df["loss"],label="loss")
plt.legend()



plt.show()
