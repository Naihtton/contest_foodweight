import platform
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, ModelCheckpoint
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

BATCH_SIZE = 20
IMAGE_SIZE = (256,384) # 2:3 ratio

def normalize_columns(df, columns_to_normalize):
    for col in columns_to_normalize:
        df[col + "_norm"] = (df[col]/300)
    return df

dataframe = pd.read_csv('fried_noodles_dataset.csv', delimiter=',', header=0)
dataframe = normalize_columns(dataframe.copy(), ['meat', 'veggie', 'noodle'])

datagen = ImageDataGenerator(
            rescale=1./255,
            brightness_range=[0.9,1.1],
            shear_range=1,
            zoom_range=0.05,
            rotation_range=10,
            width_shift_range=0.03,
            height_shift_range=0.03,
            vertical_flip=True,
            horizontal_flip=True)

datagen_noaug = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_dataframe(
    dataframe=dataframe.loc[0:1518],
    directory='images',
    x_col='filename',
    y_col=['meat_norm','veggie_norm','noodle_norm'],
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='other')

validation_generator = datagen_noaug.flow_from_dataframe(
    dataframe=dataframe.loc[1519:1703],
    directory='images',
    x_col='filename',
    y_col=['meat_norm','veggie_norm','noodle_norm'],
    shuffle=False,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='other')

test_generator = datagen_noaug.flow_from_dataframe(
    dataframe=dataframe.loc[1704:1856],
    directory='images',
    x_col='filename',
    y_col=['meat_norm','veggie_norm','noodle_norm'],
    shuffle=False,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='other')


# # Model Architecture
inputIm = Input(shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3,))
conv1 = Conv2D(64,3,activation='relu')(inputIm)
conv1 = BatchNormalization()(conv1)
pool1 = MaxPool2D()(conv1)
conv2 = Conv2D(128,3,activation='relu')(pool1)
conv2 = BatchNormalization()(conv2)
pool2 = MaxPool2D()(conv2)
conv3 = Conv2D(128, 3, activation='relu')(pool2)
conv3 = BatchNormalization()(conv3)
pool3 = MaxPool2D()(conv3)  
flat = Flatten()(pool3)
dense1 = Dense(128, activation='sigmoid')(flat)
predictedW =  Dense(3, activation='sigmoid')(dense1)


if platform.system() == "Darwin" and platform.processor() == "arm":
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)
else:
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
model = Model(inputs=inputIm, outputs= predictedW)
model.compile(optimizer=opt, loss='mse', metrics=['mean_absolute_error'])
model.summary()

class PlotLosses(Callback):
    def on_train_begin(self, logs={}):
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(epoch)
        self.losses.append(logs.get('mean_absolute_error'))
        self.val_losses.append(logs.get('val_mean_absolute_error'))

        plt.clf()
        plt.plot(self.x, self.losses, label='mean_absolute_error')
        plt.plot(self.x, self.val_losses, label='val_mean_absolute_error')
        plt.legend()
        plt.pause(0.01)
        
checkpoint = ModelCheckpoint('FriesNoodle2.h5', verbose=1, monitor='val_mean_absolute_error',save_best_only=True, mode='min')
plot_losses = PlotLosses()

#Train Model
model.fit(
    train_generator,
    steps_per_epoch= len(train_generator),
    epochs=50,
    validation_data=validation_generator,
    validation_steps= len(validation_generator),
    callbacks=[checkpoint, plot_losses])

# #Test Model
# model = load_model('FriesNoodle.h5')
# score = model.evaluate(
#     test_generator,
#     steps=len(test_generator))
# print('score (mse, mae):\n',score)

# test_generator.reset()
# predict = model.predict(
#     test_generator,
#     steps=len(test_generator),
#     workers = 1,
#     use_multiprocessing=False)
# print('prediction:\n',predict)

