from keras.models import model_from_json
import numpy as np
from ML_Project2.helpers import *

model_path = 'basic_cnn_model.json'
weights_path = "basic_cnn_weights.h5"

# load json and create model
json_file = open(model_path, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(weights_path)
print("Loaded model from disk")

loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

ids_test = np.load('skipgrams/ids_test_sg_7.npy')


prediction = loaded_model.predict(ids_test, verbose=0)

prediction[prediction >= 0.5] = 1
prediction[prediction < 0.5] = -1
prediction = prediction.reshape(-1)
print(prediction)
make_submission(prediction, 'Kaggle_prediction_10_bas_CNN.csv')
