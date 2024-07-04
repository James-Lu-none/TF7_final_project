import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import time
import matplotlib.pyplot as plt
from keras.api.layers import *
from keras.api.models import Model, Sequential
from keras.api.regularizers import l2
from keras.api.optimizers import Adam
import keras.api.backend as K

def model_structure(board_size,last_n_feature,l2_const):
    def resnext_block(inputs, filters, cardinality=32, strides=1):

        filters_per_group = filters // cardinality
        shortcut = Conv2D(filters, kernel_size=1,
                        strides=strides, padding='same')(inputs)
        shortcut = BatchNormalization()(shortcut)
        residual = []
        for _ in range(cardinality):
            x = Conv2D(filters_per_group, kernel_size=3,
                    strides=strides, padding='same')(inputs)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            residual.append(x)

        x = concatenate(residual)

        x = add([shortcut, x])
        x = Activation('relu')(x)
        return x


    in_x = network = Input((last_n_feature+4,board_size,board_size))

    network = Conv2D(64, kernel_size=3, strides=1, padding='same',kernel_regularizer=l2(l2_const))(network)
    network = BatchNormalization()(network)
    network = Activation('relu')(network)

    network = resnext_block(network, 64)
    network = resnext_block(network, 64)
    network = resnext_block(network, 64)
    network = resnext_block(network, 128)
    network = resnext_block(network, 128)
    network = resnext_block(network, 128)

    # action policy layers
    policy_net = Conv2D(filters=4, kernel_size=(1, 1), data_format="channels_first",
                        kernel_regularizer=l2(l2_const), name="policy_net_input")(network)
    policy_net = BatchNormalization()(policy_net)
    policy_net = Activation("relu")(policy_net)
    policy_net = Flatten()(policy_net)
    policy_net = Dense(board_size * board_size,
                    activation="softmax", kernel_regularizer=l2(l2_const), name="policy_net")(policy_net)
    # state value layers
    value_net = Conv2D(filters=2, kernel_size=(1, 1), data_format="channels_first",
                    kernel_regularizer=l2(l2_const), name="value_net_input")(network)
    value_net = BatchNormalization()(value_net)
    value_net = Activation("relu")(value_net)
    value_net = Flatten()(value_net)
    value_net = Dense(256, kernel_regularizer=l2(l2_const))(value_net)
    value_net = Activation("relu",)(value_net)
    value_net = Dense(128, kernel_regularizer=l2(l2_const))(value_net)
    value_net = Activation("relu",)(value_net)
    value_net = Dense(1, activation="tanh",
                    kernel_regularizer=l2(l2_const), name="value_net")(value_net)

    model = Model(in_x, [policy_net,  value_net])
    return model

if __name__ == '__main__':

    board_size = 15
    l2_const = 1e-4
    
    model = model_structure(1,board_size,l2_const)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=10000,
        decay_rate=0.9)

    optimizer = Adam(
        learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999)

    model.compile(
        optimizer=optimizer,
        loss=['categorical_crossentropy', 'mean_squared_error'],
        metrics=['accuracy']
    )

    print(model.summary())
    # net_weight = pickle.load(open("C:/Users/88692/Desktop/AlphaZero_Gomoku-master/model_record/5_30_213228_model", 'rb'))
    # model.set_weights(net_weight)