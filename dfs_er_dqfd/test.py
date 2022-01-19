import numpy as np

a = np.array([[1,3,2], [4,5,6], [7,8,9]])
b = np.array([0,2,1])
print(a[np.arange(len(a)),b])
'''
b = np.argmax(a, axis=1)
print(b)
a[np.arange(len(a)), b] = [100, 100, 100]
print((2**np.array([1,2,3,1])))
print((1-a)*0.99)
print((1-a)*0.99*[100,100,100])
print((1-a)*0.99*[100,200,100])
'''