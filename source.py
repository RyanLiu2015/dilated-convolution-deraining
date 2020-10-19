import os
import numpy as np
from PIL import Image

import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img
from keras.utils import plot_model

from utils_dark_channel import demist
from utils_image_manipulation import resize_target_image
from utils_image_manipulation import resize_single_image
from utils_image_manipulation import concatenate_images
from utils_image_manipulation import crop_image


def load_training_generator(rain_severeness='light'):

    training_rain_path = 'dataset\\' + rain_severeness + '_train\\rain\\'
    training_norain_path = 'dataset\\' + rain_severeness + '_train\\norain\\'

    i = 0

    while True:

        training_rain_array = []
        training_norain_array = []

        img_x = load_img(training_rain_path + str((i%1800)+1) + '.png')
        img_x_array = img_to_array(img_x)
        training_rain_array.append(img_x_array)

        img_y = load_img(training_norain_path + str((i%1800)+1) + '.png')
        img_y_array = img_to_array(img_y)
        training_norain_array.append(img_y_array)

        training_rain_array = np.asarray(training_rain_array)
        training_norain_array = np.asarray(training_norain_array)

        i = i + 1

        yield ([training_rain_array, training_rain_array, training_rain_array], training_norain_array)

def load_testing_generator(rain_severeness='light'):

    testing_rain_path = 'dataset\\' + rain_severeness + '_test\\rain\\'
    testing_norain_path = 'dataset\\' + rain_severeness + '_test\\norain\\'

    i = 0
    while True:

        testing_rain_array = []
        testing_norain_array = []

        img_x = load_img(testing_rain_path + str((i%200)+1) + '.png')
        img_x_array = img_to_array(img_x)
        testing_rain_array.append(img_x_array)
        

        img_y = load_img(testing_norain_path + str((i%200)+1) + '.png')
        img_y_array = img_to_array(img_y)
        testing_norain_array.append(img_y_array)

        testing_rain_array = np.asarray(testing_rain_array)
        testing_norain_array = np.asarray(testing_norain_array)

        i = i + 1

        yield ([testing_rain_array, testing_rain_array, testing_rain_array], testing_norain_array)

def load_training_group(rain_severeness='light', group_size=100):

    training_rain_path = 'dataset\\' + rain_severeness + '_train\\rain\\'
    training_norain_path = 'dataset\\' + rain_severeness + '_train\\norain\\'
    training_rain_array = []
    training_norain_array = []

    for i in range(group_size):
        img = load_img(training_rain_path + str(i+1) + '.png')
        img_array = img_to_array(img)
        training_rain_array.append(img_array)

        img = load_img(training_norain_path + str(i+1) + '.png')
        img_array = img_to_array(img)
        training_norain_array.append(img_array)

    training_rain_array = np.asarray(training_rain_array)
    training_norain_array = np.asarray(training_norain_array)
    
    return (training_rain_array, training_norain_array)

def load_testing(rain_severeness='light', group_size=20):

    testing_rain_path = 'dataset\\' + rain_severeness + '_test\\rain\\'
    testing_norain_path = 'dataset\\' + rain_severeness + '_test\\norain\\'
    testing_rain_array = []
    testing_norain_array = []

    for i in range(group_size):
        img = load_img(testing_rain_path + str(i+1) + '.png')
        img_array = img_to_array(img)
        testing_rain_array.append(img_array)

        img = load_img(testing_norain_path + str(i+1) + '.png')
        img_array = img_to_array(img)
        testing_norain_array.append(img_array)

    testing_rain_array = np.asarray(testing_rain_array)
    testing_norain_array = np.asarray(testing_norain_array)
    
    return (testing_rain_array, testing_norain_array)

def loss_function(x, y):
    loss_value = K.mean(K.sum(K.square(x-y)))
    return loss_value


def obtain_output(model, image_name):

    target_array = []

    img = load_img(image_name)

    img_array = img_to_array(img)
    target_array.append(img_array)
    target_array = np.asarray(target_array)

    output_array = model.predict([target_array, target_array, target_array])
    output_array = np.squeeze(output_array, axis=0)
    
    output_array = array_to_img(output_array)
    
    save_img('output_' + image_name, output_array)
    
    return

def derain_workflow():

    model = load_model('saved_models\\the_model.h5', custom_objects={'loss_function': loss_function})
    model.summary()

    crop_image(input_name='target.jpg')
    resize_target_image()

    concatenate_images(prefix='', output_name='step0.png')

    for i in range(9):
        obtain_output(model=model, image_name=str(i+1)+'.png')
    concatenate_images(output_name='step1.png')

    demist()

    crop_image(input_name='demist.png')
    resize_target_image()
    for i in range(9):
        obtain_output(model=model, image_name=str(i+1)+'.png')
    concatenate_images(output_name='step2.png')


    # delete intermediate images
    #for i in range(9):
        #os.remove(str(i+1) + '.png')
        #os.remove('crop_' + str(i+1) + '.png')
        #os.remove('output_' + str(i+1) + '.png')
        
    return

def derain_single_image():

    model = load_model('saved_models\\the_model.h5', custom_objects={'loss_function': loss_function})
    model.summary()

    image_name = 'target.png'

    resize_single_image(image_name=image_name)

    obtain_output(model=model, image_name='resized_' + image_name)



def train():
    
    cwd = os.getcwd()
    print('Current working directory:')
    print(cwd)

    batch_size = 20
    epochs = 20

    # input image dimensions
    img_rows, img_cols = 321, 481

    (x_train, y_train) = load_training_group()
    (x_test, y_test) = load_testing()

    x_train = x_train.astype('float32')
    x_train /= 255
    x_train = np.array(x_train)
    x_train_1 = x_train
    x_train_2 = x_train
    x_train_3 = x_train

    y_train = y_train.astype('float32')
    y_train /= 255
    y_train = np.array(y_train)

    x_test = x_test.astype('float32')
    x_test /= 255
    x_test = np.array(x_test)
    x_test_1 = x_test
    x_test_2 = x_test
    x_test_3 = x_test

    y_test = y_test.astype('float32')
    y_test /= 255
    y_test = np.array(y_test)
    
    path1_input = keras.Input(shape=(321, 481, 3))
    path1 = Conv2D(filters=4, kernel_size=(3,3), padding='same', dilation_rate=(1,1), activation='relu')(path1_input)
    path1 = Conv2D(filters=8, kernel_size=(3,3), padding='same', dilation_rate=(1,1), activation='relu')(path1)
    path1 = Conv2D(filters=8, kernel_size=(3,3), padding='same', dilation_rate=(1,1), activation='relu')(path1)

    path2_input = keras.Input(shape=(321, 481, 3))
    path2 = Conv2D(filters=4, kernel_size=(3,3), padding='same', dilation_rate=(2,2), activation='relu')(path2_input)
    path2 = Conv2D(filters=8, kernel_size=(3,3), padding='same', dilation_rate=(2,2), activation='relu')(path2)
    path2 = Conv2D(filters=8, kernel_size=(3,3), padding='same', dilation_rate=(2,2), activation='relu')(path2)

    path3_input = keras.Input(shape=(321, 481, 3))
    path3 = Conv2D(filters=4, kernel_size=(3,3), padding='same', dilation_rate=(3,3), activation='relu')(path3_input)
    path3 = Conv2D(filters=8, kernel_size=(3,3), padding='same', dilation_rate=(3,3), activation='relu')(path3)
    path3 = Conv2D(filters=8, kernel_size=(3,3), padding='same', dilation_rate=(3,3), activation='relu')(path3)

    combined = keras.layers.Average()([path1, path2, path3])
    combined = Conv2D(filters=3, kernel_size=(3,3), padding='same', dilation_rate=(1,1), activation='relu')(combined)
    combined = Conv2D(filters=3, kernel_size=(3,3), padding='same', dilation_rate=(1,1), activation='relu')(combined)
    
    model = Model(inputs=[path1_input, path2_input, path3_input], outputs=combined)

    model.compile(loss=loss_function,
                  optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2 =0.999, amsgrad=False),
                  metrics=['mean_absolute_error'])
    
    #model.fit(x=[x_train_1, x_train_2, x_train_3], y=y_train,
    #          batch_size=batch_size,
    #          epochs=epochs,
    #          verbose=1,
    #          validation_data=([x_test_1, x_test_2, x_test_3], y_test))

    model.fit_generator(load_training_generator(rain_severeness='heavy'), steps_per_epoch=1800, epochs=100, verbose=1, validation_data=load_testing_generator(rain_severeness='heavy'), validation_steps=20)


    score = model.evaluate([x_test_1, x_test_2, x_test_3], y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test mean absolute error:', score[1])

    model.save('saved_models\\model.h5')

if __name__ == '__main__':
    #crop_image(input_name='target.jpg')
    #derain_workflow()
    derain_single_image()
    #train()