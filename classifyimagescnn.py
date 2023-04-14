"""
We use tensorflow and keras to build our model.
From keras, we use the ImageDataGenerator class to import the images.
"""

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

"""We start by processing our data (images) so that we can train our model on it.

Import the training images and the testing images.
We apply feature scaling to both the training set and the test set.
Additionally, for the training set, we apply various augmentations to the images to increase the variability.
"""

train_data_gen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_data_gen = ImageDataGenerator(rescale = 1./255)

directory_to_train_images = "path" #make this the path of the training images
directory_to_test_images = "path"#make this the path of the testing images

training_set = train_data_gen.flow_from_directory(directory_to_train_images, target_size = (64, 64), batch_size = 32, class_mode = 'binary')
test_set = test_data_gen.flow_from_directory(directory_to_test_images, target_size = (64, 64), batch_size = 32, class_mode = 'binary')
# target_size specifies the pixel resolution of the images that we want to train on. Reducing this can speed up the training process.

"""We start building our model by first adding the convolutional layer."""

cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu', input_shape=[64, 64, 3]))

"""Then we do Max Pooling."""

cnn.add(tf.keras.layers.MaxPool2D(pool_size= 2 , strides = 2))

"""We add another convolutional layer and do Max Pooling."""

cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))

"""We convert the image matrix to a vector by applying flattening."""

cnn.add(tf.keras.layers.Flatten())

"""After the convolution, we connect to a fully connected layer system."""

cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))

"""Finally, we add the output layer."""

cnn.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

"""Begin training the model."""

cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

"""Making a prediction."""

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('path to test image', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image/255.0)