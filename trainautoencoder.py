from __future__ import absolute_import, division, print_function, unicode_literals


import pandas as pd
import numpy as np
import gym
import tensorflow as tf

df = pd.read_csv('observations.csv',index_col=0)
inputSize = df.columns.size
layerSize = [inputSize, inputSize/2, inputSize/6]

encoderLayers = [
    tf.keras.layers.Dense(layerSize[0],activation='relu', input_shape=(128,)),
    tf.keras.layers.Dense(layerSize[1],activation='relu'),
    tf.keras.layers.Dense(layerSize[2],activation='relu'),
    tf.keras.layers.Dense(layerSize[1],activation='relu'),
    tf.keras.layers.Dense(layerSize[0],activation='relu')
]

#decoderLayers = [
#    tf.keras.layers.Dense(layerSize[1],activation='relu'),
#    tf.keras.layers.Dense(layerSize[0],activation='relu')
#]

model = tf.keras.Sequential(encoderLayers)
tf.keras.utils.plot_model(model,'model.png',show_shapes=True)
#model.add(decoderLayers)

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss='mse',
              metrics=['mse'])

model.fit(df,df,epochs=10,batch_size=100)










