#import all necessary libraries
import tensorflow as tf
import numpy as np
import matplotlib
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

def TrainCNN():
    #generate_data() creates the Image Generator for traing and testing dataset
    def generate_data(train_path,test_path):

        TARGET_SIZE = (200, 200)
        BATCH_SIZE = 32
        CLASS_MODE = 'categorial' 
        #Creating ImageDataGenerator to load data necessary for model
        dgen_train = ImageDataGenerator(rescale=1./255,
                                        validation_split=0.2,  # using 20% of training data for validation 
                                        zoom_range=0.2,
                                        horizontal_flip=True)
        dgen_validation = ImageDataGenerator(rescale=1./255,horizontal_flip=True)
        dgen_test = ImageDataGenerator(rescale=1./255,horizontal_flip=True)


        train_generator = dgen_train.flow_from_directory(
                train_path,
                target_size=(200, 200),
                batch_size=16,
                class_mode='categorical')
        validation_generator = dgen_test.flow_from_directory(
                test_path,
                target_size=(200, 200),
                batch_size=16,
                class_mode='categorical')

        train_data = train_generator
        valid_data = validation_generator
        return train_generator,validation_generator

    n = 0 #global variable
    def number_class():
        n_classes=int(input("Enter number of classes in the dataset:"))
        n = n_classes
        return n_classes

    n = number_class()


    #Define CNN Model Architecture 

    def model_architecture():
        model = Sequential()
        model.add(Conv2D(32, (5,5), padding='same', activation='relu',input_shape=(200, 200, 3)))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(32, (5,5), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (5,5), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(64, (5,5), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (5,5), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(128, (5,5), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(n, activation='softmax'))
        model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    train_generator,validation_generator = generate_data('dataset/train','dataset/test')

    model = model_architecture()

    def declare_checkpoint(path):

        checkpoint_path = path
        checkpoint_dir = os.path.dirname(checkpoint_path)

        model_checkpoint_callback = ModelCheckpoint(
                filepath=checkpoint_path,
                save_weights_only=True,
                monitor='val_loss',
                mode='min', 
                save_best_only=False)

        return model_checkpoint_callback
    
    checkpoints_folder=declare_checkpoint("model_checkpoints/")

    history = model.fit(
            train_generator,
            batch_size = 16,
            epochs=50,
            validation_data=validation_generator, callbacks = [checkpoints_folder],
            validation_freq=1)
    return history
    
    
TrainCNN()
