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

# a = np.arange(1,7)
# print(a)
# print(a.shape)
# b = a[np.newaxis, :]
# c = a[:, np.newaxis]
# print(c)
# print(b)
# print(b.shape)

#concatenation

# a = np.array([[1,2], [3,4]])
# print(a)
# b = np.array([[5,6]])
# c = np.concatenate((a,b.T), axis=1) #need to transpose when using 1, can also do axis=None
# print(c)

#stacking arrays
# a = np.array([1,2,3,4])
# b = np.array([5,6,7,8])
# #hstack
# c = np.vstack((a,b))
# print(c)

# broadcasting
# x = np.array([[1,2,3], [4,5,6], [1,2,3], [4,5,6]])
# a = np.array([1,0,1],[1,0,1], [1,0,1], [1,0,1])
# y = x + a
# print(y)

# a = np.array([[7,8,9,10,11,12,13], [17,18,19,20,21,22,23]])
# print(a)
# print(a.sum(axis=1)) # 1 sum entry for each row, 0 axis would be one sum over each column, can also use None
# print(a.var(axis=None)) #variance
# print(a.std(axis=None)) #standard deviation

# datatypes
# x = np.array([1.0,2.0], dtype=np.int64) #can pass in dtype to specify data type
# print(x)
# print(x.dtype)

# # copying arrays
# a = np.array([1,2,3])
# b = a
# b[0] = 42
# print(b)
# print(a) # only copied reference, and both objects point to same memory location, so modifies both arrays. To copy use a.copy

# a = np.zeros((2,3))
# print(a)

# a = np.ones((2,3))
# print(a)

# a = np.full((2,3), 5.0)
# print(a)

# a = np.eye(3)
# print(a)

# a = np.arange(20)
# print(a)

# a = np.linspace(0,10,5)
# print(a)

#random numbers
# a = np.random.random((3,2)) # generate 3 x 2 array with values between 0 and 1
# print(a)

# a = np.random.randn(1000) # normal / Gaussian
# print(a.mean(), a.var())

# a = np.random.randint(3,10, size=(3,3))
# print(a)

# linear algebra
# eigenvalues PCA (principle component analysis)

# a = np.array([[1,2], [3,4]])
# eigenvalues, eigenvectors = np.linalg.eig(a)

# # print(eigenvalues)
# # print(eigenvectors) # column vector - col 0 corresponds to eigenvalue 0

# # e_vec * e_val = A * e_vec (formula to verify)
# b = eigenvectors[:,0] * eigenvalues[0]
# print(b)
# c = a @ eigenvectors[:,0]
# print(b)

# # compare values -- compare if 2 arrays are equal use np all close function
# print(np.allclose(b,c))

# solving linear systems 

# A = np.array([[1,1], [1.5, 4.0]])
# b = np.array([2200, 5050])

# # not a great way, slow and has numerical issues
# # x = np.linalg.inv(A).dot(b) # inverse of A times b, dot product of A and b
# # print(x)

# # better way same result but faster
# x = np.linalg.inv(A).dot(b)
# print(x)

# x = np.linalg.solve(A, b)
# print(x)

# load CSV files
# np.loadtxt, np.genfromtxt (either one works)
data = np.loadtxt('spambase.csv', delimiter=",", dtype=np.float32)
print(data.shape)




