# conding = UTF-8

import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dropout, Activation, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.initializers import he_normal 
from keras import optimizers
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.data_utils import get_file

batch_size = 128
epochs = 164
iterations = 391
num_classes = 10
dropout = 0.5
weight_decay = 0.0005
log_filepath  = './vgg19_retrain_logs/'

# def color_preprocessing(x_train, x_test):
#     x_train = (x_train.astype('float32') - 128) / 128
#     x_test = (x_test.astype('float32') - 128) / 128
#     return x_train, x_test

def color_preprocessing(x_train,x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [123.680, 116.779, 103.939]
    # std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:,:,:,i] = x_train[:,:,:,i] - mean[i]
        x_test[:,:,:,i] = x_test[:,:,:,i] - mean[i]
    return x_train, x_test

def conv_layer(filters, kernel_size, name, input_shape=None):
    params = {
        "filters": filters,
        "kernel_size": kernel_size,
        "padding": 'same',
        'kernel_regularizer': keras.regularizers.l2(weight_decay),
        'kernel_initializer': he_normal(),
        'name': name
    }
    if input_shape is not None:
        params['input_shape'] = input_shape
    return Conv2D(**params)

def build_model():
    model = Sequential()

    # Block 1
    model.add(conv_layer(64, 3, 'block1_conv1', x_train.shape[1:]))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(conv_layer(64, 3, 'block1_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(conv_layer(128, 3, 'block2_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(conv_layer(128, 3, 'block2_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(conv_layer(256, 3, 'block3_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(conv_layer(256, 3, 'block3_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(conv_layer(256, 3, 'block3_conv3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(conv_layer(256, 3, 'block3_conv4'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(conv_layer(512, 3, 'block4_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(conv_layer(512, 3, 'block4_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(conv_layer(512, 3, 'block4_conv3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(conv_layer(512, 3, 'block4_conv4'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))


    # Block 5
    model.add(conv_layer(512, 3, 'block5_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(conv_layer(512, 3, 'block5_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(conv_layer(512, 3, 'block5_conv3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(conv_layer(512, 3, 'block5_conv4'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # model modification for cifar-10
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, use_bias = True, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='fc_cifa10'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(4096, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='fc2'))  
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))      
    model.add(Dense(10, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='predictions_cifa10'))        
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    # load pretrained weight from VGG19 by name      
    model.load_weights(filepath, by_name=True)

    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

def scheduler(epoch):
    learning_rate_init = 0.08
    if epoch >= 81:
        learning_rate_init = 0.01
    if epoch >= 122:
        learning_rate_init = 0.001
    return learning_rate_init

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
filepath = get_file('vgg19_weights_tf_dim_ordering_tf_kernels.h5', WEIGHTS_PATH, cache_subdir='models')

# load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train, x_test = color_preprocessing(x_train, x_test)


if __name__ == '__main__':
    # build network
    model = build_model()
    print(model.summary())

    # set callback
    tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr,tb_cb]

    # set data augmentation
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        width_shift_range=0.125,
        height_shift_range=0.125,
        fill_mode='constant',
        cval=0.)
    datagen.fit(x_train)

    # start training
    model.fit_generator(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=iterations,
        epochs=epochs,
        callbacks=cbks,
        validation_data=(x_test, y_test))
    model.save('retrain.h5')
