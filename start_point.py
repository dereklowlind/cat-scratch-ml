from fastbook import *
import time
# from fastai.vision.widgets import *

learn_inf = load_learner('export.pkl')

img1 = 'test_images/ns1.jpg'

start = time.time()
print(learn_inf.predict('test_images/ns1.jpg'))

print(learn_inf.predict('test_images/ns2.jpg'))
print(learn_inf.predict('test_images/s1.jpg'))
print(learn_inf.predict('test_images/s2.jpg'))


end = time.time()
print("The time of execution of above program is :", (end-start)*1000, " milliseconds")