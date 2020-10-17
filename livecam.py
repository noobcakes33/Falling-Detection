import cv2
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
import numpy as np
import time

# model = keras.models.load_model('CNN_model/')

def load_cnn_model():
    # load json and create model
    json_file = open('model_v5.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model_v5.h5")
    print("Loaded model from disk")
    print("Model V3 Initializaing...")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return loaded_model

loaded_model = load_cnn_model()
cap = cv2.VideoCapture("fall_vids/20200810_161536.mp4")

while True:
    ret, frame = cap.read()
    original_frame = frame
    font = cv2.FONT_HERSHEY_SIMPLEX

    if ret:
        # img=image.load_img("Dataset/Validation/non_falling/"+i, target_size=(150, 150))
        # img = cv2.imread(frame)
        # x = image.img_to_array(img)
        original_frame = cv2.resize(original_frame, (512,512))
        frame = cv2.resize(frame, (224,224)) ## 150, 150 (v2)
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        original_frame = cv2.rotate(original_frame, cv2.ROTATE_90_CLOCKWISE)
        x = np.expand_dims(frame, axis=0)
        date = time.localtime()

        yr, mon, day, hr, min, sec = date[:6]
        classes = loaded_model.predict(x)
        if int(classes[0][0]) == 0:
            print("[Activity] Fall")
            cv2.putText(original_frame, 'FALL', (10, 60), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0,0,255), 2)
            cv2.putText(original_frame, str(yr)+"-"+str(mon)+"-"+str(day), (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,0,0), 2)
            cv2.putText(original_frame, str(hr)+":"+str(min)+":"+str(sec), (380, 20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,0,0), 2)

        else:
            print("[Activity] Non_Fall")
            cv2.putText(original_frame, 'NON_FALL', (10, 60), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0,255,0), 2)
            cv2.putText(original_frame, str(yr)+"-"+str(mon)+"-"+str(day), (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,0,0), 2)
            cv2.putText(original_frame, str(hr)+":"+str(min)+":"+str(sec), (380, 20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,0,0), 2)

        cv2.imshow('frame',original_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
