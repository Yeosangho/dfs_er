import numpy as np
from collections import deque

a= np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
arr = []
arr.append(a[..., np.newaxis])
arr.append(b[..., np.newaxis])

x = arr[::-1]
print(np.concatenate(x, axis=2)[0,0,0] )
print(np.concatenate(x, axis=2)[0,1,0] )
print(np.concatenate(x, axis=2)[1,0,0] )
print(np.concatenate(x, axis=2)[1,1,0] )

print( 2 != 1 & 0 > 1)

a = deque(maxlen=10)
a.append(1.2)
a.append(2.1)
print(str(sum([t for i, t in enumerate(a)])))
for i in range(4):
	print(i)
	if(i == 2):
		break;