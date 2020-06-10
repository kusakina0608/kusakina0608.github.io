---
layout: post
title:  "백준 17284 Vanding Machine"
date:   2020-04-24 14:11:00 +0900
author: kusakina0608
tags:
  - 백준
  - Input
categories: [Algorithm]
---



[백준 17284 - Vending Machine](https://www.acmicpc.net/problem/17284)



이 문제에서는 입력 개수가 주어지지 않고 바로 입력이 주어진다.

입력 개수를 먼저 입력받고, 지정된 횟수만큼 반복문을 돌리는 것과 다르게 입력을 받아야 한다.

[코드](http://boj.kr/c6056dbff5d34560acc75dd08ba2316b)

``` C++
int main(void){
    int 상품총액 = 0;
    int 상품번호 = 0;
    do {
        cin>>상품번호;
        상품총액 += 상품가격[상품번호];
    } while (getc(stdin) == ' ');
    cout<<(소지금-상품총액);
    return 0;
}
```



그리고, 일반적으로 속도가 느린 c++의 iostream 속도를 빠르게 해 주기 위해서

```C++
ios_base::sync_with_stdio(0);
cin.tie(0);
```

를 사용하는 경우가 많은데, ios::sync_with_stdio(false); 를 한 뒤에는

iostream 함수와 stdio 함수를 같이 쓰면 안된다.