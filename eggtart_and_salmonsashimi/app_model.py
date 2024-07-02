
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping ,ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import time 

def image_gen_w_aug(train_parent_directory, valid_parent_directory, test_parent_directory):
    train_datagen = ImageDataGenerator(rescale=1/255,
                                       rotation_range=30,  
                                       zoom_range=0.2, 
                                       width_shift_range=0.1,  
                                       height_shift_range=0.1)
    
    test_datagen = ImageDataGenerator(rescale=1/255)
    val_datagen = ImageDataGenerator(rescale=1/255)
    
    train_generator = train_datagen.flow_from_directory(train_parent_directory,
                                                        target_size=image_size,
                                                        batch_size=batch_size,
                                                        class_mode='categorical')
    
    val_generator = val_datagen.flow_from_directory(valid_parent_directory,
                                                    target_size=image_size,
                                                    batch_size=batch_size,
                                                    class_mode='categorical')
    
    test_generator = test_datagen.flow_from_directory(test_parent_directory,
                                                      target_size=image_size,
                                                      batch_size=batch_size,
                                                      class_mode='categorical')
    
    return train_generator, val_generator, test_generator

def model_output_for_TL(pre_trained_model, last_output):
    x = Conv2D(16, (3, 3), activation='relu')(last_output)
    x = MaxPooling2D(pool_size=(2, 2), strides=1)(x)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=1)(x)
    
    x = Flatten()(x)
    
    # # First Fully Connected Layer
    # x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
    # x = Dropout(0.5)(x)

    # # Second Fully Connected Layer
    # x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    # x = Dropout(0.5)(x)
    
    # # Third Fully Connected Layer
    # x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    # x = Dropout(0.5)(x)
    
    # Output Layer
    x = Dense(3, activation='softmax',kernel_regularizer=l2(l2_reg))(x)
    
    model = Model(pre_trained_model.input, x)
    
    return model

train_dir = os.path.join('./datasets/train')
test_dir = os.path.join('./datasets/test')
valid_dir = os.path.join('./datasets/valid')

image_size = (150, 150)
batch_size = 64
epoch = 50

learning_rate = 1e-4
l2_reg = 0.001


train_generator, validation_generator, test_generator = image_gen_w_aug(train_dir, valid_dir, test_dir)

pre_trained_model = InceptionV3(input_shape=(150, 150, 3), 
                                include_top=False, 
                                weights='imagenet')

for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output

model_TL = model_output_for_TL(pre_trained_model, last_output)
model_TL.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                 loss='categorical_crossentropy', 
                 metrics=['accuracy'])

steps_size = train_generator.samples // train_generator.batch_size

# early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor = 0.5, patience = 3 , min_lr=1e-7)
# start time
start_time = time.time()

# training model
history_TL = model_TL.fit(
    train_generator,
    steps_per_epoch=steps_size,  
    epochs=epoch,
    verbose=1,
    validation_data=validation_generator,
    callbacks=[lr_scheduler,early_stopping]
)

# end time
end_time = time.time()
duration = end_time - start_time

print('Total time taken to train the model: ' + str(duration))

tf.keras.models.save_model(model_TL, 'my_model.hdf5')

# Plot the accuracy
plt.figure()

# Accuracy Plot
plt.plot(history_TL.history['accuracy'], label='Training Accuracy')
plt.plot(history_TL.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# Plot the loss
plt.figure()

# Loss Plot
plt.plot(history_TL.history['loss'], label='Training Loss')
plt.plot(history_TL.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.show()


