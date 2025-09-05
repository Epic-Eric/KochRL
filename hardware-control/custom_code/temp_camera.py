import cv2
import time

camera = cv2.VideoCapture(2)
time.sleep(2)
camera.set(cv2.CAP_PROP_FPS, 30)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
print(camera.get(cv2.CAP_PROP_FRAME_HEIGHT), camera.get(cv2.CAP_PROP_FRAME_WIDTH), camera.get(cv2.CAP_PROP_FPS))
if not camera.isOpened():
    raise Exception("Could not open video device")
while True:
    ret, frame = camera.read()
    while not ret:
        ret, frame = camera.read()
    frame = cv2.flip(frame, 0)
    cv2.imshow("Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
