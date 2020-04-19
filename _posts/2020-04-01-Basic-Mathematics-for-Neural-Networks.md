---
layout: single
title:  "Basic Mathematics for Neural Networks"
date:   2020-04-01 12:00:00 +0900
tags:
  - Artificial Neural Network
  - Linear Algebra
  - Math
  - Numpy
categories: "Artificial-Neural-Network"
---

## 벡터 덧셈(Vector Addition)

import package

```python
import re, math, random
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from functools import partial, reduce
import numpy as np
```

Create data

```python
v = [x for x in range(1, 11, 2)]
w = [y for y in range(11, 21, 2)]
```

Python ver. Vector Addition

```python
def vector_add(v, w):
    return [v_i + w_i for v_i, w_i in zip(v,w)]
vector_add(v, w)
```

Numpy ver. Vector Addition

```py
np.array(v) + np.array(w)
```



## Python arithmetic vs. Numpy arithmetic

Create data

```python
v = [x for x in range(1, 11, 2)]
w = [y for y in range(11, 21, 2)]
```

Python ver. 

```python
%timeit vector_add(v, w)
```

Numpy ver.

```python
%timeit np.array(v) + np.array(w)
```



## 벡터 뺄셈(Vector Subtraction)

 Create data

```python
v = [x for x in range(1, 11, 2)]
w = [y for y in range(11, 21, 2)]
```

Python ver. Vector Subtraction

```python
def vector_subtract(v, w):
    return [v_i - w_i for v_i, w_i in zip(v,w)]
vector_subtract(v, w)
```

Numpy ver. Vector Subtraction

```python
np.array(v) - np.array(w)
```



## 벡터 리스트 덧셈

Create data

```python
v = [x for x in range(1, 11, 2)]
w = [y for y in range(11, 21, 2)]
vectors = [v,w,v,w,v,w]
```

Python ver. List of Vector Addition 1

```python
def vector_sum(vectors):
    return reduce(vector_add, vectors)
vector_sum(vectors)
```

Python ver. List of Vector Addition 2

```python
def vector_sum_modified(vectors):
    return [sum(value) for value in zip(*vectors)]
vector_sum_modified(vectors)
```

Numpy ver. List of Vector Addition

```python
np.sum(vectors, axis=0)
```



## 벡터 스칼라 곱(Multiply a Vector by a Scalar)

Create data

```python
v = [x for x in range(1, 11, 2)]
scalar = 3
```

Python ver. Multiply a Vector by a Scalar

```python
def scalar_multiply(c, v):
    return [c * v_i for v_i in v]
scalar_multiply(scalar, v)
```

Numpy ver. Multiply a Vector by a Scalar

```python
scalar * np.array(v)
```



## 벡터 리스트 평균(Means of a List of Vector)

Create data

```python
v = [1,2,3,4]
w = [-4,-3,-2,-1]
```

Python ver. Means of a List of Vector

```python
def vector_mean(vectors):
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))
vector_mean([v,v,v,v])
```

Numpy ver. Means of a List of Vector

```python
np.mean([v,v,v,v], axis=0)
```



## 백터의 내적(Vector Dot Product)

Create data

```pytho
v = [1,2,3,4]
w = [-4,-3,-2,-1]
```

Python ver. Vector Dot Product

```python
def dot(v, w):
    return sum(v_i * w_i for v_i, w_i in zip(v, w))
dot(v, w)
```

Numpy ver. Vector Dot Product

```python
np.dot(v,w)
```



## 벡터 성분 제곱 값의 합(Sum of Squares)

Create data

```python
v = [1,2,3,4]
```

Python ver. Sum of Squares

```python
def sum_of_squares(v):
    return dot(v, v)
sum_of_squares(v) # v * v = [1,4,9,16]
```

Python ver. Magnitude (or length)

```python
def magnitude(v):
    return math.sqrt(sum_of_squares(v))
magnitude(v)
```

Numpy ver. Magnitude

```python
np.linalg.norm(v)
```



## 두 벡터 사이의 거리(Distance Between Two Vectors)

Create data

```python
v = [1,2,3,4]
w = [-4,-3,-2,-1]
```

Python ver. Squared dist.

```python
def squared_distance(v, w):
    return sum_of_squares(vector_subtract(v, w))
squared_distance(v,w) 
```

Python ver. Euclidean Distance

```python
def distance(v, w):
    return math.sqrt(squared_distance(v, w))
distance(v,w)
```

Numpy ver. Euclidean Distance 1

```python
np.linalg.norm(np.subtract(v,w))
```

Numpy ver. Euclidean Distance 2

```python
np.sqrt(np.sum(np.subtract(v,w)**2))
```



## 행렬 형태(Matrices)

Create data

```python
example_matrix = [[1,2,3,4,5], [11,12,13,14,15], [21,22,23,24,25]]
example_matrix_np = np.array(example_matrix)
```

Python ver. Shape

```python
def shape(A):
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols
shape(example_matrix)
```

Python ver. Get row

```python
def get_row(A, i):
    return A[i]
get_row(example_matrix, 0)
```

Python ver. Get Column

```python
def get_column(A, j):
    return [A_i[j] for A_i in A]
get_column(example_matrix,3)
```

Numpy ver. Shape

```python
np.shape(example_matrix)
```

Numpy ver. Get Row(Slicing)

```python
example_matrix_np[0]
```

Numpy ver. Get Column(Slicing)

```python
example_matrix_np[:,3] 
```



## 행렬 생성(Matrix Generation)

Python ver. Matrix Generation

```python
def make_matrix(num_rows, num_cols, entry_fn):
    return [[entry_fn(i, j) for j in range(num_cols)]
            for i in range(num_rows)]
def is_diagonal(i, j):
    return 1 if i == j else 0
identity_matrix = make_matrix(5, 5, is_diagonal)
identity_matrix
```

Numpy ver. Matrix Generation

```python
np.identity(5)
```



## 이진 관계(Binary Relationship)

```python
friendships = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]
friendships =  [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0], 	# user 0
                [1, 0, 1, 1, 0, 0, 0, 0, 0, 0], 	# user 1
                [1, 1, 0, 1, 0, 0, 0, 0, 0, 0], 	# user 2
                [0, 1, 1, 0, 1, 0, 0, 0, 0, 0], 	# user 3
                [0, 0, 0, 1, 0, 1, 0, 0, 0, 0], 	# user 4
                [0, 0, 0, 0, 1, 0, 1, 1, 0, 0], 	# user 5
                [0, 0, 0, 0, 0, 1, 0, 0, 1, 0], 	# user 6
                [0, 0, 0, 0, 0, 1, 0, 0, 1, 0], 	# user 7
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 1], 	# user 8
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]] 	# user 9
friends_of_five = [i for i, is_friend in enumerate(friendships[5]) if is_friend]
print(friends_of_five)
```



## 행렬 덧셈(Matrix Addition)

Create data

```python
A = [[ 1., 0., 0.], [ 0., 1., 2.]]
B = [[ 5., 4., 3.], [ 2., 2., 2.]]
```

Python ver. Matrix Addition

```python
def matrix_add(A, B):
    if shape(A) != shape(B):
        raise ArithmeticError("cannot add matrices with different shapes")
    num_rows, num_cols = shape(A)
    def entry_fn(i, j): return A[i][j] + B[i][j]
    return make_matrix(num_rows, num_cols, entry_fn)
matrix_add(A,B)
```

Numpy ver. Matrix Addition

```python
np.add(A,B)
```



## 벡터 점곱 그래프(Matrix Dot Product)

```python
def make_graph_dot_product_as_vector_projection(plt):
    v = [2, 1]
    w = [math.sqrt(.25), math.sqrt(.75)]
    c = dot(v, w)
    vonw = scalar_multiply(c, w)
    o = [0,0]

    plt.arrow(0, 0, v[0], v[1], 
              width=0.002, head_width=.1, length_includes_head=True)
    plt.annotate("v", v, xytext=[v[0] + 0.1, v[1]])
    plt.arrow(0 ,0, w[0], w[1], 
              width=0.002, head_width=.1, length_includes_head=True)
    plt.annotate("w", w, xytext=[w[0] - 0.1, w[1]])
    plt.arrow(0, 0, vonw[0], vonw[1], length_includes_head=True)
    plt.annotate(u"(v•w)w", vonw, xytext=[vonw[0] - 0.1, vonw[1] + 0.1])
    plt.arrow(v[0], v[1], vonw[0] - v[0], vonw[1] - v[1], 
              linestyle='dotted', length_includes_head=True)
    plt.scatter(*zip(v,w,o),marker='.')
    plt.axis([0,2,0,2]) # 잘리는 부분이 있어서 변경
    plt.show()
%matplotlib inline
make_graph_dot_product_as_vector_projection(plt)
```



## 행렬 점곱 (Matrix Dot Product)

Create data

```python
A = [[ 1., 2., 3.],
     [ 1., 2., 3.],
     [ 1., 2., 3.]]
B = [[ 1., 2., 3.],
     [ 1., 2., 3.],
     [ 1., 2., 3.]]
```

Python ver. Inner Product

```python
def my_matrix_dot(A, B):
    return [[dot(A[i], [B_i[j] for B_i in B]) for j in range(len(A))]
            for i in range(len(B[0]))]
```

```python
for M in my_matrix_dot(A, B):
    print(M)
```

Numpy ver. Inner Product

```python
np.dot(A,B)
```



## 전치 행렬(Transpose Matrix)

Create data

```python
A = [[ 1., 2., 3.],
     [ 4., 5., 6.]]
B = [[ 5., 4., 3.],
     [ 2., 2., 2.]]
```

Python ver. Transpose Matrix

```python
def my_matrix_transpose(M):
    return [[M[j][i] for j in range(len(M))]
            for i in range(len(M[0]))]
```

```python
for M in my_matrix_transpose(A):
    print(M)
```

```python
for M in my_matrix_transpose(B):
    print(M)
```

Numpy ver. Transpose Matrix

```python
np.transpose(A)
np.transpose(B)
```

