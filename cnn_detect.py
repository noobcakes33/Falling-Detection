import cv2
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
import numpy as np


# model = keras.models.load_model('CNN_model/')

def load_cnn_model():
    # load json and create model
    json_file = open('model_v2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model_v2.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return loaded_model

loaded_model = load_cnn_model()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    original_frame = frame
    font = cv2.FONT_HERSHEY_SIMPLEX

    if ret:
        # img=image.load_img("Dataset/Validation/non_falling/"+i, target_size=(150, 150))
        # img = cv2.imread(frame)
        # x = image.img_to_array(img)

        frame = cv2.resize(frame, (150,150))
        x = np.expand_dims(frame, axis=0)

        classes = loaded_model.predict(x)
        if int(classes[0][0]) == 0:
            print("[Activity] Fall")
            cv2.putText(original_frame, 'FALL', (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0,0,255), 2)

        else:
            print("[Activity] Non_Fall")
            cv2.putText(original_frame, 'NON_FALL', (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0,255,0), 2)

        cv2.imshow('frame',original_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
