from keras.models import model_from_json
import numpy as np
from helpers import *

model_path = 'keras_files/crnn6_model.json'
weights_path = "keras_files/crnn6_weights.h5"

# load json and create model
json_file = open(model_path, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(weights_path)
print("Loaded model from disk")

loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

ids_test = np.load('skipgrams/ids_test_sg_2.npy')

print(ids_test[:4])
fake_ids = []
for i in range(len(ids_test)):
    fake_ids.append(ids_test[i].tolist())

print(fake_ids[:4])
prediction = loaded_model.predict(fake_ids, verbose=0)

prediction[prediction >= 0.5] = 1
prediction[prediction < 0.5] = -1
prediction = prediction.reshape(-1)
print(prediction)
make_submission(prediction, 'Crnn_6')
