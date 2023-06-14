import numpy as np
import sys

path = './zh_en/'
fn = 'timeSource.npy'

timeSource = np.load(path + fn, allow_pickle=True)
#print(embeding[0], embeding[1], embeding[4000], embeding[4001])


#for i in range(100):
#    test = timeSource[i]
#    print(test[0], test[1])
#python test_readVec.py

timeSource = timeSource.tolist()
print(len(timeSource))
print(sys.getsizeof(timeSource))
for i in range(20):
    test = timeSource.popitem()
    if len(test[1]) < 200:
        print(test[0], test[1], '\n\n')
    else:
        print('\n\n')
