---
layout: single
title:  "Divide and Conquer"
date:   2020-04-03 12:00:00 +0900
tags:
  - Algorithm
  - Divide and Conquer
categories: Algorithm
---

## 분할정복 알고리즘(Divide and Conquer)

분할 정복 알고리즘은 그대로 해결할 수 없는 문제를 작은 문제로 분할하여 문제를 해결하는 방법이나 알고리즘이다. 빠른 정렬이나 합병 정렬로 대표되는 정렬 알고리즘 문제와 고속 푸리에 변환 문제가 대표적이다.

### 1. 합병 정렬

**분할 과정**

37 10 22 30 35 13 25 24

37 10 22 30 / 35 13 25 24

37 10 / 22 30 / 35 13 / 25 24

37 / 10 / 22 / 30 / 35 / 13 / 25 / 24

**병합 과정**

37 / 10 / 22 / 30 / 35 / 13 / 25 / 24

10 37 / 22 30 / 13 35 / 24 25

10 22 30 37 / 13 24 25 35

10 13 22 24 25 30 35 37



### 2. 퀵 정렬

퀵 정렬은 n개의 데이터를 정렬할 때, 최악의 경우에는 O(n2)번의 비교를 수행하고, 평균적으로 O(n log n)번의 비교를 수행한다.

1. 리스트 가운데서 하나의 원소를 고른다. 이렇게 고른 원소를 피벗이라고 한다.
2. 피벗 앞에는 피벗보다 값이 작은 모든 원소들이 오고, 피벗 뒤에는 피벗보다 값이 큰 모든 원소들이 오도록 피벗을 기준으로 리스트를 둘로 나눈다. 이렇게 리스트를 둘로 나누는 것을 분할이라고 한다. 분할을 마친 뒤에 피벗은 더 이상 움직이지 않는다.
3. 분할된 두 개의 작은 리스트에 대해 재귀(Recursion)적으로 이 과정을 반복한다. 재귀는 리스트의 크기가 0이나 1이 될 때까지 반복된다.



## 과제

스트라센 알고리즘 조사해서 정리하기



### 3. 최근접 점쌍 구하기

x좌표값(또는 y좌표값)을 기준으로 전체를 두개의 그룹으로 쪼갬

서브그룹에서 최근접 점쌍 탐색

분할선을 기준으로 최근접 점쌍의 길이 범위 내에서 최근접 점쌍 탐색

![Closest-Fair-01](https://github.com/kusakina0608/kusakina0608.github.io/blob/master/assets/images/2020-04-03-Divide-and-Conquer/Closest-Fair-01.gif?raw=true)

![Closest-Fair-01](https://github.com/kusakina0608/kusakina0608.github.io/blob/master/assets/images/2020-04-03-Divide-and-Conquer/Closest-Fair-02.gif?raw=true)

![Closest-Fair-03](https://github.com/kusakina0608/kusakina0608.github.io/blob/master/assets/images/2020-04-03-Divide-and-Conquer/Closest-Fair-03.gif?raw=true)

