import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras as keras
import tensorflow_hub as hub

print(f"TF: {tf.__version__}")
print(f"Keras: {keras.__version__}")
print(f"Hub: {hub.__version__}")

class MyCallback(keras.callbacks.Callback):
    def __init__(self, test_data):
        super(MyCallback, self).__init__()
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs=None):
        print(f"\n--------- pre-predict stop_training={self.model.stop_training}\n")
        #The problem is in the prediction: if commented ES works fine
        predictions = self.model.predict(self.test_data.batch(512))
        print(f"\n--------- post-predict stop_training={self.model.stop_training}\n")
        logs['my_metric'] = 1

train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews", 
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

#print(len(list(test_data)))

hub_layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1",#"https://tfhub.dev/google/nnlm-en-dim50/2", 
                           output_shape=50,
                           input_shape=[], 
                           trainable=True, 
                           dtype=tf.string)

model = keras.Sequential()
model.add(hub_layer)
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-3), #force overfit faster
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
print(model.summary())

es = keras.callbacks.EarlyStopping(patience=2)
myc = MyCallback(test_data)

#This causes EarlyStop not to stop
my_callbacks = [es, myc]
#This one works fine
#my_callbacks = [myc, es]
#my_callbacks = [es]

model.fit(train_data.batch(512), 
          validation_data=validation_data.batch(512),
          epochs=100, 
          callbacks=my_callbacks, 
          verbose=1)



