import numpy as np

from keras.preprocessing.image import ImageDataGenerator

class Preprocess():
    def __init__(self, batch_size):
        self.batch_size = batch_size


    def makeGenerator(self, fit_val):
        train_datagen = ImageDataGenerator(rotation_range     = fit_val['rotation_range'],\
                                   horizontal_flip            = fit_val['horizontal_flip'],\
                                   vertical_flip              = fit_val['vertical_flip'],\
                                   rescale                    = 1 / 255.,\
                                   fill_mode                  = "constant", \
                                   cval                       = 0,\
                                   width_shift_range          = fit_val['width_shift_range'], \
                                   height_shift_range         = fit_val['height_shift_range'], \
                                   zoom_range                 = fit_val['zoom_range'],\
                                   brightness_range           = fit_val['brightness_range']\
                                   )
        return train_datagen

    def makePredictGenerator(self):
        return ImageDataGenerator(rescale = 1/ 255.)
        