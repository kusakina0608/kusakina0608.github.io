---
layout: post
title: "Strassen Algorithm"
date:   2020-04-11 12:00:00 +0900
author: kusakina0608
use_math: true
tags:
  - Math
  - Strassen Algorithm
categories: [Algorithm]
---
## Strasse의 행렬곱셈

일반적인 정방행렬의 행렬곱셈은 O(n^3)의 성능을 갖는다.

Strassen의 행렬곱셈 알고리즘을 사용하면 행 수가 n=2^k인 행렬에 대해 O(n^2.81)의 성능으로 곱셈을 수행한다. 



## Divide and Conquer로 행렬 곱셈하기

행렬 A와 B를 곱할 때, 주어진 행렬 A와 B를 분할하여 부분행렬로 나눈다.

부분행렬에 대해 곱셈 연산을 적용한다.



$$
\begin{pmatrix}
A_1 & A_2\\
A_3 & A_4\\
\end{pmatrix}
\begin{pmatrix}
B_1 & B_2\\
B_3 & B_4\\
\end{pmatrix}
=
\begin{pmatrix}
C_1 & C_2\\
C_3 & C_4\\
\end{pmatrix}
\\
C_1=A_1B_1+A_2B_3\\
C_2=A_1B_2+A_2B_4\\
C_3=A_3B_1+A_4B_3\\
C_4=A_3B_2+A_4B_4\\
$$


이런 방식으로 DNC를 적용하여 행렬곱을 수행하면 성능에는 변화가 없다.

행렬 덧셈은 곱셈에 비해 훨씬 적은 시간이 소요되며, 위와 같은 행렬의 곱셈은

8번의 행렬 곱셈, 4번의 행렬 덧셈이 이루어지므로 시간복잡도는 아래와 같다.


$$
T(n)=O(n^{log_28})=O(n^3)
$$


## Strassen의 행렬 곱셈

Strassen의 행렬 곱셈은 아래와 같이 수행된다.


$$
\begin{pmatrix}A_1 & A_2\\A_3 & A_4\\\end{pmatrix}\begin{pmatrix}B_1 & B_2\\B_3 & B_4\\\end{pmatrix}=\begin{pmatrix}C_1 & C_2\\C_3 & C_4\\\end{pmatrix}\\
P=(A_1+A_4)(B_1+B_4)\\
Q=(A_3+A_4)B_1\\
R=A_1(B_2-B_4)\\
S=A_4(B_3-B_1)\\
T=(A_1+A_2)B_4\\
U=(A_3-A_1)(B_1+B_2)\\
V=(A_2-A_4)(B_3-B_4)\\
C_1=P+S-T+V\\
C_2=R+T\\
C_3=Q+S\\
C_4=P+R-Q+U\\
$$


DNC 행렬곱셈에서와 동일하게 시간복잡도를 계산해보기 위해 연산의 횟수를 계산해보면

Strassen의 행렬곱셈에서는 7번의 행렬 곱셈, 18번의 행렬 덧셈이 이루어진다. 


$$
T(n)=O(n^{log_27})=O(n^{2.81})\\
$$


Strassen의 행렬 곱셈은 일반적인 행렬 곱셈에 비해 엄청난 성능 증가는 볼 수 없다.

Strassen의 행렬 곱셈의 효과를 보려면 매우 큰 행렬이어야 한다.

만약 그렇지 않을 경우, 많은 덧셈횟수와 프로그램의 복잡성 등의 오버헤드로 인해 오히려 느리게 동작할 수도 있다.

