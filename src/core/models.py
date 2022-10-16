import tensorflow as tf
from keras import backend as K
from keras.layers.core import Dense, Lambda, Flatten, Dropout,Reshape
from keras.layers.pooling import MaxPooling2D
from keras.layers import Convolution2D, Cropping2D, Input, merge, UpSampling2D,GRU, Activation,BatchNormalization
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, LearningRateScheduler
from keras.models import Sequential, load_model, Model
from keras.optimizers import Adam
from src.core.losses import ctc_lambda_func

def build_small_UNet(img_size, channels):
    
    inputs = Input((img_size[1], img_size[0], channels), name = 'Input')
    inputs_norm = Lambda(lambda x: x/255 - 1., name = 'Normalization')
    conv1 = Convolution2D(8, 3, 3, activation='relu', border_mode='same', name = 'Conv1_1')(inputs)
    conv1 = Convolution2D(8, 3, 3, activation='relu', border_mode='same', name = 'Conv1_2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name = 'MaxPool1')(conv1)

    conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', name = 'Conv2_1')(pool1)
    conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', name = 'Conv2_2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name = 'MaxPool2')(conv2)

    conv3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name = 'Conv3_1')(pool2)
    conv3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name = 'Conv3_2')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name = 'MaxPool3')(conv3)

    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name = 'Conv4_1')(pool3)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name = 'Conv4_2')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), name = 'MaxPool4')(conv4)

    conv5 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name = 'Conv5_1')(pool4)
    conv5 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name = 'Conv5_2')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2), name = 'Upsample2D_1')(conv5), conv4], mode='concat', concat_axis=3, name = 'Merge1')
    conv6 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name = 'Conv6_1')(up6)
    conv6 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name = 'Conv6_2')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2), name = 'Upsample2D_2')(conv6), conv3], mode='concat', concat_axis=3, name = 'Merge2')
    conv7 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name = 'Conv7_1')(up7)
    conv7 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name = 'Conv7_2')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2), name = 'Upsample2D_3')(conv7), conv2], mode='concat', concat_axis=3, name = 'Merge3')
    conv8 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', name = 'Conv8_1')(up8)
    conv8 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', name = 'Conv8_2')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2), name = 'Upsample2D_4')(conv8), conv1], mode='concat', concat_axis=3, name = 'Merge4')
    conv9 = Convolution2D(8, 3, 3, activation='relu', border_mode='same', name = 'Conv9_1')(up9)
    conv9 = Convolution2D(8, 3, 3, activation='relu', border_mode='same', name = 'Conv9_2')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid', name = 'Output')(conv9)

    model = Model(input=inputs, output=conv10)

    
    return model

def build_classification_model(keep_prob,channels,img_height,img_width,categories):
        
    model = Sequential()   
   # model.add(Lambda(lambda x: (x / 255.0) - 0.5, name = 'Normalization', input_shape = (img_height,img_width,channels)))   

    #model.add(BatchNormalization(input_shape=(img_height,img_width,channels), axis=1))
    model.add(Convolution2D(32, 1, 1, input_shape = (img_height,img_width,channels) , init = 'normal', activation = 'elu', border_mode = 'same', subsample = (1,1), 
                            bias=True, name = 'Conv1'))
    model.add(MaxPooling2D(pool_size=(2, 2), name = 'MaxPool1'))   
     
    model.add(Convolution2D(64, 1, 1, init = 'normal', activation = 'elu', border_mode = 'same', subsample = (1,1), 
                            bias=True, name = 'Conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name = 'MaxPool2'))
       
    
    model.add(Convolution2D(128, 1, 1, init = 'normal', activation = 'elu', border_mode = 'same', subsample = (1,1), 
                            bias=True, name = 'Conv3'))
    model.add(MaxPooling2D(pool_size=(2, 2), name = 'MaxPool3'))
       
      
    model.add(Convolution2D(256, 1, 1, init = 'normal', activation = 'elu', border_mode = 'same', subsample = (1,1), 
                            bias=True, name = 'Conv4'))
    model.add(MaxPooling2D(pool_size=(2, 2), name = 'MaxPool4'))  
 
    model.add(Convolution2D(512, 1, 1, init = 'normal', activation = 'elu', border_mode = 'same', subsample = (1,1), 
                            bias=True, name = 'Conv5'))
    model.add(MaxPooling2D(pool_size=(2, 2), name = 'MaxPool5'))   
      
    model.add(Flatten(name = 'Flatten'))   
      
    model.add(Dense(500, activation = 'elu', name = 'FC1'))
    model.add(Dropout(keep_prob, name = 'Dropout1'))    
    model.add(Dense(500, activation = 'elu', name = 'FC2'))
    model.add(Dropout(keep_prob, name = 'Dropout2'))   
    model.add(Dense(categories,activation='softmax' , name = 'Output'))  

    return model    

def build_engine(alphabet, size, drop_prob):
    
    conv_filters = 16
    kernel_size = 3
    pool_size = 2
    time_dense_size = 64 #32
    rnn_size = 512
    act = 'relu'
    input_dim = (size[0], size[1], 1) #Image width (n_cols) is size[0], image height (n_rows) is size[1]. Fed backwards.
    max_plate_chars = 6
     
    #Setup input and normalize
    inputs = Input(input_dim, name = 'Input')
    labels = Input(name='Labels', shape=[max_plate_chars], dtype='float32')
    input_length = Input(name='Input_length', shape=[1], dtype='int64')
    label_length = Input(name='Label_length', shape=[1], dtype='int64')
    
    inputs_norm = Lambda(lambda x: x/255 - 0.5, output_shape = input_dim, name = 'Normalization')(inputs)
    #Define 2 convolutional layers with maxpoolin
    conv1_1 = Convolution2D(conv_filters, kernel_size, kernel_size, border_mode='same', activation=act, 
                   name='Conv1_1')(inputs_norm)
    conv1_2 = Convolution2D(conv_filters, kernel_size, kernel_size, border_mode='same', activation=act, 
                   name='Conv1_2')(conv1_1)
    maxpool1 = MaxPooling2D(pool_size=(pool_size, pool_size), name='Maxpool1')(conv1_2)
    
    conv2_1 = Convolution2D(2*conv_filters, kernel_size, kernel_size, border_mode='same', activation=act, 
                   name='Conv2_1')(maxpool1)
    conv2_2 = Convolution2D(2*conv_filters, kernel_size, kernel_size, border_mode='same', activation=act, 
                   name='Conv2_2')(conv2_1)
    
    maxpool2 = MaxPooling2D(pool_size=(pool_size, pool_size), name='Maxpool2')(conv2_2)
    conv_to_rnn_dims = (size[0] // (pool_size ** 2), (size[1] // (pool_size ** 2)) * 2 * conv_filters)
    reshape = Reshape(target_shape=conv_to_rnn_dims, name='Reshape')(maxpool2)
    
    #conv_to_rnn_dims = (size[0] // (pool_size), (size[1] // (pool_size)) * 2 * conv_filters)
    #reshape = Reshape(target_shape=conv_to_rnn_dims, name='Reshape')(conv2_2)

    #Cuts down input size going into RNN:
    dense1 = Dense(time_dense_size, activation=act, name='Dense1')(reshape)
    
    #Setup two layers of bi-directional GRU units
    gru_1 = GRU(rnn_size, return_sequences=True, init='he_normal', name='GRU1_a', 
                dropout_W = drop_prob, dropout_U =drop_prob)(dense1)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='GRU1_b', 
                dropout_W = drop_prob, dropout_U =drop_prob)(dense1)
    gru1_merged = merge([gru_1, gru_1b], mode='sum', name = 'Add')
    gru_2 = GRU(rnn_size, return_sequences=True, init='he_normal', name='GRU2_a', 
                dropout_W = drop_prob, dropout_U =drop_prob)(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='GRU2_b', 
                dropout_W = drop_prob, dropout_U =drop_prob)(gru1_merged)
    dense2 = Dense(len(alphabet)+1, init='he_normal',name='Dense2')(merge([gru_2, gru_2b], mode='concat', name='Concatenate'))
    y_pred = Activation('softmax', name='softmax')(dense2)
    
    #Define the loss output
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([labels, y_pred, input_length, label_length])

    model = Model(input=[inputs, labels, input_length, label_length], output=loss_out)
    
    return model