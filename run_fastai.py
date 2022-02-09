from fastbook import *
import cv2
from playsound import playsound
from queue import Queue

vid = cv2.VideoCapture(0)
# vid = cv2.VideoCapture(2)

learn_inf = load_learner('export.pkl')

log = open("run_log.txt", "a")
rolling_average = 0
num_rolling = 10

q = Queue()
for i in range(num_rolling):
    q.put(0)

while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
  
    # Display the resulting frame
    # cv2.imshow('frame', frame)
    results = learn_inf.predict(frame)
    print(results)
    scratch_pred = float(results[2][1])
    rolling_average -= q.get()
    rolling_average += scratch_pred
    q.put(scratch_pred)
    print(rolling_average/num_rolling)
    if rolling_average/num_rolling > 0.6:
        log.write('scratching pred:' + str(rolling_average) +' \n')
        playsound('good_kitty.mp3')
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

