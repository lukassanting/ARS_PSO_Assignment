
import numpy as np

arr = np.array([[1,2],[3,4]])
arr2 = np.array([[5,6],[7,8]])
arr3 = np.array(['a','b'])

print(arr[0][1])

for i in range(len(arr[0])):
    if(arr3[i] == 'b'):
        arr[:,i] = arr2[:,i]

print(arr)

