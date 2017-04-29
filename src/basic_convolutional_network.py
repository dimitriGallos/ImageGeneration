from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Deconvolution2D, Flatten
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from preprocessing import load_data
from preprocessing import mask
from preprocessing import save
from preprocessing import crop
import numpy as np
original_dim = 64*64*3

#Build the network
input_img = Input(shape=(64, 64,3))

x = Convolution2D(64, 5, 5, activation='relu', border_mode='same',dim_ordering='tf')(input_img)
x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D(pool_size=(2, 2), border_mode='same')(x)
x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D(pool_size=(2, 2), border_mode='same')(x)


x = Convolution2D(256,5,5, activation='relu', border_mode='same')(encoded)
x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode = 'same' )(x)


autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='sgd', loss='mean_squared_error')

training_data, testing_data = load_data()
y_train = training_data['data']
y_test = testing_data['data']
x_train = np.copy(y_train)
x_test = np.copy(y_test)
x_train = mask(x_train)
x_test = mask(x_test)
y_train = crop(y_train)


es = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, verbose=1)
checkpoint = ModelCheckpoint('../models/test', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
autoencoder.fit(x_train, y_train,
                nb_epoch=50,
                batch_size=100,
                shuffle=True,
                validation_split= 0.3, verbose = 1)

output = autoencoder.predict(x_test,100)
autoencoder.save('test2.h5')
save(output,y_test, dir = '../resultsEncoderDecoderCropped')