---
layout: post
title:  "Tizen Overview"
date:   2020-04-02 12:00:00 +0900
author: kusakina0608
tags:
  - Tizen
categories: [Tizen]
---

## Tizen Overview

Linux, Linux mobile, 삼성전자, 인텔이 공동으로 개발한 오픈 소스 모바일 운영체제.

스마트폰, 스마트 워치, TV, 냉장고 같은 다양한 스마트 기기에서 작동하는 표준 기반의 모바일 운영체제.

스마트 기기를 임베디드 시스템 개발자, 웹 개발자들이 유연하게 사용할 수 있도록 고안됨.

- Native API는 C언어를 이용한 임베디드 시스템 소프트웨어로 개발하기에 유용.
- Web API는 HTML5, CSS, Javascript를 사용하여 간단한 어플리케이션을 만들 수 있도록 함.

소프트웨어 개발 키트(SDK)를 통해 각종 tool과 API를 제공.



## Tizen profile

다양한 임베디드 디바이스 종류와 목적에 맞는 profile들을 제공.
Tizen 3.0부터 3가지의 profile type을 제공

- 스마트폰
- 스마트 워치
- TV



## Tizen application types

Tizen platform은 Linux 커널을 기반으로 하며, 3가지의 application type을 제공.

![Tizen Architecture](https://github.com/kusakina0608/kusakina0608.github.io/blob/master/assets/images/2020-04-02-Tizen/tizen-architecture.png?raw=true)

* Native application

Native application은 Native API를 사용하며, Linux C를 기반으로 메모리와 성능의 이점이 있음.

스마트폰, 스마트 워치에서 사용하는 하드웨어들에 인터페이스를 제공.
카메라, GPS, 센서 등과 같은 디바이스들에 더 쉽게 접근할 수 있도록 API 제공.

스마트폰, 스마트 워치 개발을 지원.

EFL(Enlightenment Foundation Libraries)

EDC(Edje Data Collection)

* Web application

TAU(Tizen Advanced UI): Tizen에 맞는 다양한 형태의 UI component 들을 만들 수 있도록 제공(Button, slider와 같은 시각적인 UI 요소들을 표현, 스마트 워치에 맞는 circular 형태의 UI component 들을 포함)

* Hybrid application(Native + Web)

Native service application과 Web application을 하나의 패키지로 하여 유용한 application 개발 가능

Service를 통해 Native subsystems의 활용과 Web application을 통해 UI 개발이 가능(Application 간 통신을 활용)

* Tizen .NET application

TV, 스마트폰, 스마트 워치, IoT 기기들에서 실행되는 Tizen application을 개발할 수 있도록 새롭게 고안됨

Native application과 Web application의 단점들을 보완하여 고안됨(Native application은 runtime을 관리하기 어려움, Web application은 Native application에 비해 제한된 기능만을 사용할 수 있고, 성능에 문제 발생)

총 4가지의 프로그램 개발 환경을 제공

.NET Standard API: .NET core의 가장 주요한 요소 중 하나인 .NET Standard를 사용.

Xamarin.Forms: C#, XAML을 사용하는 cross-platform API를 제공.

Tizen.Wearable.CircularUI: Tizen wearable에 특화된 circular UI를 제공.

TizenFX API: Tizen platform에 특화된 디바이스 기능을 사용할 수 있도록 제공.