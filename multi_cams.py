import cv2
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
import numpy as np

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

cap1 = cv2.VideoCapture("fall_detection_v3/vids/1.mp4")
cap2 = cv2.VideoCapture("fall_detection_v3/vids/2.mp4")
cap3 = cv2.VideoCapture("fall_detection_v3/vids/3.mp4")

while(cap1.isOpened() | cap2.isOpened() | cap3.isOpened()):
    # Capture frame-by-frame
    ret, frame1 = cap1.read()
    ret, frame2 = cap2.read()
    ret, frame3 = cap3.read()

    original_frame1 = frame1
    original_frame2 = frame2
    original_frame3 = frame3

    font = cv2.FONT_HERSHEY_SIMPLEX

    if ret == True:
        frame1 = cv2.resize(frame1, (224,224))
        frame2 = cv2.resize(frame2, (224,224))
        frame3 = cv2.resize(frame3, (224,224))

        x1 = np.expand_dims(frame1, axis=0)
        x2 = np.expand_dims(frame2, axis=0)
        x3 = np.expand_dims(frame3, axis=0)

        classes1 = loaded_model.predict(x1)
        if int(classes1[0][0]) == 0:
            print("[CAM 1 Activity] Fall")
            cv2.putText(original_frame1, 'FALL', (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0,0,255), 2)

        else:
            print("[CAM 1 Activity] Non_Fall")
            cv2.putText(original_frame1, 'NON_FALL', (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0,255,0), 2)

        classes2 = loaded_model.predict(x2)
        if int(classes2[0][0]) == 0:
            print("[CAM 2 Activity] Fall")
            cv2.putText(original_frame2, 'FALL', (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0,0,255), 2)

        else:
            print("[CAM 2 Activity] Non_Fall")
            cv2.putText(original_frame2, 'NON_FALL', (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0,255,0), 2)


        classes3 = loaded_model.predict(x3)
        if int(classes3[0][0]) == 0:
            print("[CAM 3 Activity] Fall")
            cv2.putText(original_frame3, 'FALL', (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0,0,255), 2)

        else:
            print("[CAM 3 Activity] Non_Fall")
            cv2.putText(original_frame3, 'NON_FALL', (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0,255,0), 2)

        # Display the resulting frame
        cv2.imshow('CAM 1', original_frame1)
        cv2.imshow('CAM 2', original_frame2)
        cv2.imshow('CAM 3', original_frame3)


    # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break


cap1.release()
cap2.release()

cv2.destroyAllWindows()
