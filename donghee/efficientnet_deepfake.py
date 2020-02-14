from keras.layers import GlobalAveragePooling2D, Dropout, Dense, concatenate, Conv2D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.models import Sequential, Model
from keras.optimizers import Adam
from math import ceil
import keras.backend.tensorflow_backend as K
import tensorflow as tf
import efficientnet.keras as efn 
import numpy as np
import argparse
import random
import keras
import sys
import cv2
import csv
import os

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))
    

class EFNet():
    def __init__(self, output_size, batch_size, ef_nm, face_shape, lip_shape,audio_shape, sample_size):
        self.output_size = output_size
        self.batch_size = batch_size
        self.ef_nm = ef_nm
        self.weight_path = "weight/"
        self.log_path = "log/" + self.ef_nm + ".csv"
        self.face_shape = face_shape
        self.lip_shape = lip_shape
        self.audio_shape = audio_shape
        self.sample_size = sample_size
        
        self.seed = 5


    def getB4Net(self, shape, model_name):
        effnet = efn.EfficientNetB4(weights=None,\
                                include_top=False,\
                                input_shape=shape)
        effnet.load_weights(self.weight_path + 'efficientnet-b4_imagenet_1000_notop.h5')

        for i, layer in enumerate(effnet.layers):
            effnet.layers[i].name = str(model_name) + "_" + layer.name
            if "batch_normalization" in layer.name:
                effnet.layers[i] = GroupNormalization(groups=self.batch_size, axis=-1, epsilon=0.00001)
        return effnet


    def getB3Net(self, shape, model_name):
        effnet = efn.EfficientNetB3(weights=None,\
                                include_top=False,\
                                input_shape=shape)
        effnet.load_weights(self.weight_path + 'efficientnet-b3_imagenet_1000_notop.h5')

        for i, layer in enumerate(effnet.layers):
            effnet.layers[i].name = str(model_name) + "_" + layer.name
            if "batch_normalization" in layer.name:
                effnet.layers[i] = GroupNormalization(groups=self.batch_size, axis=-1, epsilon=0.00001)
        return effnet
    
    def getB2Net(self, shape, model_name):
        effnet = efn.EfficientNetB2(weights=None,\
                                include_top=False,\
                                input_shape=shape)
        effnet.load_weights(self.weight_path + 'efficientnet-b2_imagenet_1000_notop.h5')

        for i, layer in enumerate(effnet.layers):
            effnet.layers[i].name = str(model_name) + "_" + layer.name
            if "batch_normalization" in layer.name:
                effnet.layers[i] = GroupNormalization(groups=self.batch_size, axis=-1, epsilon=0.00001)
        return effnet    

    
    def buildModel(self):
        model_face = Sequential()
        model_face.add(self.getB4Net(self.face_shape, "face_eff"))
        model_face.add(GlobalAveragePooling2D())
        model_face.add(Dropout(0.1))
        model_face.add(Dense(10, activation='relu')) 
        
        model_lip = Sequential()
        model_lip.add(self.getB3Net(self.lip_shape, "lip_eff"))
        model_lip.add(GlobalAveragePooling2D())
        model_lip.add(Dropout(0.1))
        model_lip.add(Dense(10, activation='relu')) 
        
        model_voice = Sequential()
        model_voice.add(Conv2D(32,2,activation='relu', kernel_initializer = 'he_normal', input_shape=self.audio_shape, data_format="channels_last"))
        #model_voice.add(Conv2D(32,1,activation='relu', kernel_initializer = 'he_normal', data_format="channels_last"))
        
        #model_voice.add(Dense(20, activation='relu'))
        #model_voice.add(Dense(20, activation='relu'))
        model_voice.add(Flatten())
        #model_voice.add(GlobalAveragePooling2D())
        model_voice.add(Dropout(0.1))
        model_voice.add(Dense(10, activation='relu')) 
        

        concat_model = concatenate([model_face.output, model_lip.output, model_voice.output])
        
        concat_output = Dense(1, activation='sigmoid')(concat_model)
        
        result_model = Model([model_face.input, model_lip.input, model_voice.input], [concat_output])
        result_model.compile(loss="binary_crossentropy",\
                      optimizer=Adam(lr=0.00005),\
                      metrics=['acc'])
        self.model = result_model


    def focal_loss(self, gamma=2., alpha=4.):
        gamma = float(gamma)
        alpha = float(alpha)
        def focal_loss_fixed(y_true, y_pred):
            epsilon = 1.e-9
            y_true = tf.convert_to_tensor(y_true, tf.float32)
            y_pred = tf.convert_to_tensor(y_pred, tf.float32)
            model_out = tf.add(y_pred, epsilon)
            ce = tf.multiply(y_true, -tf.log(model_out))
            weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
            fl = tf.multiply(alpha, tf.multiply(weight, ce))
            reduced_fl = tf.reduce_max(fl, axis=1)
            return tf.reduce_mean(reduced_fl)
        return focal_loss_fixed


    def fitGenerator(self, \
                     generator,\
                     train_face, \
                     train_lip, \
                     train_audio, \
                     train_label, \
                     val_face, \
                     val_lip, \
                     val_audio, \
                     val_label, \
                     train_len, \
                     val_len):
        
        es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=12)
        weight_nm = self.weight_path + self.ef_nm + "-{epoch:02d}-loss_{loss:.4f}-valloss_{val_loss:.4f}-valacc_{val_acc:.4f}.h5"
        checkpoint = ModelCheckpoint(weight_nm, \
                                     monitor='val_acc', \
                                     verbose=1, \
                                     save_best_only=True, \
                                     save_weights_only=True, \
                                     mode='max')
        #csvlogger = CSVLogger(self.log_path, append=True, separator=',')
        self.model.fit_generator(self.generator_three_data(generator, train_face, train_lip, train_audio, train_label), \
                                validation_data=self.generator_three_data(generator, val_face, val_lip, val_audio, val_label), \
                                validation_steps=(val_len * self.sample_size) // self.batch_size, \
                                steps_per_epoch=(train_len * self.sample_size) // self.batch_size, \
                                epochs=1, \
                                callbacks=[es, checkpoint])
        self.save()

    def generator_three_data(self, generator, face_data, lip_data, audio_data, label_data):
        gen_x_face = generator.flow(face_data, label_data, batch_size=self.batch_size, seed=self.seed)
        gen_x_lip = generator.flow(lip_data, label_data, batch_size=self.batch_size, seed=self.seed)
        gen_x_audio = generator.flow(audio_data, label_data, batch_size=self.batch_size, seed=self.seed)
        
        while True:
            x_face = gen_x_face.next()
            x_lip = gen_x_lip.next()
            x_audio = gen_x_audio.next()
            
            yield [x_face[0], x_lip[0], x_audio[0]], x_face[1]


    def save(self):
        self.model.save_weights(self.weight_path+self.ef_nm+".h5")
        print("success save model weights")


    def load(self):
        self.model.load_weights(self.weight_path+self.ef_nm+".h5")
        print("success load model weights")


    def predict(self, test_x):
        return self.model.predict(test_x)

    def predictGenerator(self, predict_generator, pred_face, pred_lip, pred_audio, img_len):
        return self.model.predict_generator(self.generator_three_data_P(predict_generator,pred_face, pred_lip, pred_audio), steps=ceil(img_len/self.batch_size))


    def generator_three_data_P(self, generator, face_data, lip_data, audio_data):
        gen_x_face = generator.flow(face_data, batch_size=self.batch_size, shuffle=False)
        gen_x_lip = generator.flow(lip_data, batch_size=self.batch_size, shuffle=False)
        gen_x_audio = generator.flow(audio_data, batch_size=self.batch_size, shuffle=False)
        
        while True:
            x_face = gen_x_face.next()
            x_lip = gen_x_lip.next()
            x_audio = gen_x_audio.next()
            
            yield [x_face, x_lip, x_audio]
