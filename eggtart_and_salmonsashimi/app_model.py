# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 04:09:59 2020

@author: ASUS
"""

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import os
import matplotlib.pyplot as plt

def image_gen_w_aug(train_parent_directory,valid_parent_directory, test_parent_directory):
    
    train_datagen = ImageDataGenerator(rescale=1/255,
                                      rotation_range = 30,  
                                      zoom_range = 0.2, 
                                      width_shift_range=0.1,  
                                      height_shift_range=0.1,
                                      validation_split = 0.15)
    
    test_datagen = ImageDataGenerator(rescale=1/255)
    
    train_generator = train_datagen.flow_from_directory(train_parent_directory,
                                                       target_size = (150, 150),
                                                       batch_size = 214,
                                                       class_mode = 'categorical'
                                                       )
    
    val_generator = train_datagen.flow_from_directory(valid_parent_directory,
                                                          target_size = (150, 150),
                                                          batch_size = 37,
                                                          class_mode = 'categorical'
                                                          )
    
    test_generator = test_datagen.flow_from_directory(test_parent_directory,
                                                     target_size=(150, 150),
                                                     batch_size = 37,
                                                     class_mode = 'categorical')
    
    return train_generator, val_generator, test_generator


def model_output_for_TL (pre_trained_model, last_output):
    
    x = Conv2D(64, (3, 3), activation='relu',kernel_regularizer=l2(0.001))(last_output)
    x = MaxPooling2D(pool_size=(2, 2), strides=1)(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_regularizer=l2(0.001))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=1)(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_regularizer=l2(0.001))(x)

    x = Flatten()(x)
    
    # First Fully Connected Layer
    x = Dense(512, activation='relu',kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)

    # Second Fully Connected Layer
    x = Dense(256, activation='relu',kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    
    # Third Fully Connected Layer
    x = Dense(128, activation='relu',kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    
    # Output Layer
    x = Dense(3, activation='softmax')(x)
    
    model = Model(pre_trained_model.input, x)
    
    return model


train_dir = os.path.join('./datasets/train')
test_dir = os.path.join('./datasets/test')
valid_dir= os.path.join('./datasets/valid')


train_generator, validation_generator, test_generator = image_gen_w_aug(train_dir,valid_dir,test_dir)

pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                include_top = False, 
                                weights = 'imagenet')

for layer in pre_trained_model.layers:
  layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed3')
last_output = last_layer.output

model_TL = model_output_for_TL(pre_trained_model, last_output)
model_TL.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#early stopping to prevent overfiting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)


history_TL = model_TL.fit(
      train_generator,
      steps_per_epoch=10,  
      epochs=50,
      verbose=1,
      validation_data = validation_generator,
      callbacks=[early_stopping]
      
)



tf.keras.models.save_model(model_TL,'my_model.hdf5')

# Plotting the loss and accuracy
acc = history_TL.history['accuracy']
val_acc = history_TL.history['val_accuracy']
loss = history_TL.history['loss']
val_loss = history_TL.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 12))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model_TL.summary()

