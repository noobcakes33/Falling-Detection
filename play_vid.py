import numpy as np
import cv2

vid_name = "20200810_163938.mp4"
cap = cv2.VideoCapture(f"fall_vids/{vid_name}")
count = 1



while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (224,244))
    #gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(f"frames/{vid_name}_frame{count}.jpg", gray)
    # Display the resulting frame
    cv2.imshow('frame',gray)
    print("[frame] ", count)
    count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
