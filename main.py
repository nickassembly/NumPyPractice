import numpy as np

# l1 = [1,2,3]
# l2 = [4,5,6]
# a1 = np.array(l1)
# a2 = np.array(l2)

# # Dot Products

# # traditional way
# dp = 0
# for i in range(len(l1)):
#     dp += l1[i] * l2[i]
# print(dp)

# # numpy way
# dp = np.dot(a1,a2)
# print(dp)

# dp = a1 @ a2
# print(dp)

# # multidimensional arrays
# a = np.array([[1,2], [3,4]])
# print(a)
# print(a.shape)

# print(a[0,0])
# print(a[:,0])
# print(a[0,:])

# print(np.linalg.inv(a)) #calc inverse
# print(np.linalg.det(a)) #calc determinate

# c = np.diag(a)
# print(np.diag(c)) # calc diagonal

# # slicing
# print(a)

# b = a[0,1:3] 
# #b1 = a[a:,1] # only column 1
# print(b)

# a2 = np.array([[1,2], [3,4], [5,6]])
# print(a2)

# bool_idx = a2 > 2
# print(bool_idx)
# print(a2[bool_idx])

# bb = np.where(a>2, a, -1)
# print(bb)

# fancy indexing
# a = np.array([10,19,30,41,50,61])
# print(a)
# b = [1,3,5]
# print(a[b])

# print(a)
# even = np.argwhere(a%2==0).flatten()
# print(a[even])

# reshaping arrays

