import cv2

# 20200810_101051.mp4 done

vid_name = "20200810_101333.mp4"
vidcap = cv2.VideoCapture(f'fall_vids/{vid_name}')

ret, image = vidcap.read()
count = 1

while ret:
  cv2.imwrite(f"frames/{vid_name}_frame{count}.jpg", image)
  success,image = vidcap.read()

  print("[frame] ", count)
  count += 1
