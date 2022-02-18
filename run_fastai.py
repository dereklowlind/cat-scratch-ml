from fastbook import *
import cv2
from playsound import playsound
from queue import Queue
import time
import serial

ser = serial.Serial('/dev/ttyACM6', 9600) # Define the serial port and baud rate.
# vid = cv2.VideoCapture(0) # onboard webcam
vid = cv2.VideoCapture(2) # usb webcam
num_rolling = 10
predict_threshold = 0.9
between_scratch_threshold = 10 # in seconds

learn_inf = load_learner('export.pkl')

log = open("run_log.txt", "a")
rolling_average = 0
last_scratch = time.time() - between_scratch_threshold - 1 

q = Queue()
for i in range(num_rolling):
    q.put(0)

while(True):
    ret, frame = vid.read()
    # cv2.imshow('frame', frame)
    results = learn_inf.predict(frame)
    print(results)
    scratch_pred = float(results[2][1])
    rolling_average -= q.get()
    if rolling_average < 0:
        rolling_average = 0
    rolling_average += scratch_pred
    q.put(scratch_pred)
    print(rolling_average/num_rolling)
    if rolling_average/num_rolling > predict_threshold and (time.time()-last_scratch) > between_scratch_threshold:
        log.write('scratching pred:' + str(rolling_average/num_rolling) +' \n')
        ser.write(b'D')
        playsound('good_kitty.mp3', False)
        rolling_average = 0
        last_scratch = time.time()
    
    # the 'q' button is set as the quitting button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

