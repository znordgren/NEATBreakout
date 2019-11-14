from __future__ import absolute_import, division, print_function, unicode_literals


import pandas as pd
import numpy as np
import gym
import tensorflow as tf

df = pd.read_csv('observations.csv',index_col=0)

dataset = tf.data.Dataset.from_tensor_slices((df.values,df.values))
dataset = dataset.shuffle(len(df)).batch(1)

inputSize = df.columns.size
layerSize = [inputSize, inputSize/2, inputSize/4]

encoderLayers = [
    tf.keras.layers.Dense(layerSize[0],activation='relu', input_shape=(inputSize,)),
    tf.keras.layers.Dense(layerSize[1],activation='relu'),
    tf.keras.layers.Dense(layerSize[2],activation='relu'),
    tf.keras.layers.Dense(layerSize[1],activation='relu'),
    tf.keras.layers.Dense(layerSize[0],activation='relu')
]

model = tf.keras.Sequential(encoderLayers)
tf.keras.utils.plot_model(model,'model.png',show_shapes=True)
#model.add(decoderLayers)

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss='mse',
              metrics=['mae'])

model.fit(dataset,epochs=6)

model.summary()

subDF = df.sample(n=10,replace=True,random_state=1)
print(model.predict(subDF))
model.save('fullae.h5')

model.pop()
model.pop()

print(model.predict(subDF))

model.save('ae.h5')











