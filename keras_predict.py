from keras.models import model_from_json
import numpy as np
from helpers import *

model_path = 'keras_files/crnn2_model.json'
weights_path = "keras_files/crnn2_weights.h5"

# load json and create model
json_file = open(model_path, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(weights_path)
print("Loaded model from disk")

loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

ids_test = np.load('ids_test.npy')


prediction = loaded_model.predict(ids_test, verbose=0)

prediction[prediction >= 0.5] = 1
prediction[prediction < 0.5] = -1
prediction = prediction.reshape(-1)
print(prediction)
make_submission(prediction, 'Crnn_2')
