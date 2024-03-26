from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pandas as pd

def normalize_columns(df, columns_to_normalize):
    for col in columns_to_normalize:
        df[col] = (df[col]/300)
    return df

BATCH_SIZE = 20
IMAGE_SIZE = (256,384) # 2:3 ratio

dataframe = pd.DataFrame(columns=['filename', 'meat', 'veggie', 'noodle'])
for i in range(1, 301):
    dataframe = dataframe._append({'filename': "{}.jpg".format(i), 'meat': 0.0, 'veggie': 0.0, 'noodle': 0.0}, ignore_index=True)
dataframe = normalize_columns(dataframe, ['meat', 'veggie', 'noodle'])
datagen = ImageDataGenerator(rescale=1./255)

test_generator = datagen.flow_from_dataframe(
    dataframe=dataframe.loc[0:300],
    directory='ranking_round/images',
    x_col='filename',
    y_col=["meat","veggie","noodle"],
    shuffle=False,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='other')

model = load_model('FriesNoodle2.h5')

score = model.evaluate(
    test_generator,
    steps=len(test_generator))
print('score (mse, mae):\n',score)

test_generator.reset()
predict = model.predict(
    test_generator,
    steps=len(test_generator),
    workers = 1,
    use_multiprocessing=False)
print('prediction:\n',predict)

for index in dataframe.index:
    dataframe.loc[index, 'meat'] = int(predict[index][0] * 300)
    dataframe.loc[index, 'veggie'] = int(predict[index][1] * 300)
    dataframe.loc[index, 'noodle'] = int(predict[index][2] * 300)

# Save to CSV
dataframe.to_csv('FriesNoodle2_result.csv', index=False) 