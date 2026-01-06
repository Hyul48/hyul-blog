---
title: "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis(cont...)"
date: 2026-01-06
draft: false
math: true
---

# NeRF : Representing Scenes as Neural Radiance Fields for View Synthetic

## Executive Summary
- NeRF는 장면을 연속적인 5D 함수로 표현한다: 입력은 **3D 위치 x**와 **시선 방향 d**, 출력은 **볼륨 밀도 σ**와 **색 c**이다.
- 픽셀은 카메라에서 쏜 광선을 따라 여러 점을 샘플링 하고 각 점에서 $(\sigma, c)$를 얻어 **볼륨 렌더링(확률적 누적)**으로 색을 합성한다.
- $\sigma$는 그 지점에서 광선이 종료(충돌)할 확률의 미분형으로 해석되고, 투과율은 $T$는 지금까지 아무것도 안 맞고 살아남았을 확률로 해석된다.
- 연속 적분은 이산 샘플 합으로 근사되며, 이 형태는 **미분 가능**해서 픽셀 MSE 손실이 네트워크 파라미터 $\Theta$로 역전파 된다.
- 효율을 위해 coarse 결과로 만든 가중치 $\omega_i$를 PDF(확률 밀도 함수)로 써서, 중요한 구간에 fine 샘플을 더 배치하는 **hierarchical smapling**을 활용한다.

## 1. Background & Problem Setting
### 1.1 Novel View Synthesis 문제
- 입력 : 한 장면을 여러 각도에서 찍은 이미지들 + 각 이미지의 카메라 파라미터(포즈 포함)
- 목표 : 훈련 때 없었던 새로운 카메라 위치/자세에서의 이미지를 그럴듯하게 생성(렌더링)

여기서 핵심 포인트는 "장면을 무엇으로 저장할거냐" 인데 NeRF는 그 **저장방식을 격자(voxel)나 메시(mesh)가 아니라 함수로 택함**

### 1.2 NERF의 관점
- 장면을 "3D 모델 파일"로 저장하는 대신 위치/방향에 따라 보이는 빛은 무엇인가를 답하는 **연속 함수**로 저장

    $\rightarrow$ 이 관점 때문에, 한 번 학습되면 어떤 카메라에서도 광선만 쏘면 이미지를 뽑을 수 있다.(=view synthesis)
## 2. Sceneepresenatation as a Neural Field
### 2.1 5D 입력과 출력의 정의 (NeRF의 함수)
Nerf는 MLP로 다음 함수를 근사한다:
- 입력 : $(x,d)$
    - $x = (x,y,z)$: 공간 위치(3D)
    - $d$ :시선 방향 (보통 $\theta, \phi$ 또는 단위 벡터로 표현)

- 출력 : ($\sigma$, c)
    - $\sigma(x)$ : 볼륨 밀도
    - $ c(x,d)$ : 그 방향에서 보이는 RGB 색

### 2.2 왜 $\sigma$는 방향을 안보고, c는 방향을 보나?
- $\sigma$는 공간에 물체가 존재하냐(기하/점유)에 가까움 $\rightarrow$ 방향에 따라 달라지면 3D가 흔들림
- 색 c는 재질/반사 때문에 시선 방향에 따라 달라질 수 있음(view-dependent)
즉, 기하(geometry)와 외관(appearance)을 분리하는 최소한의 귀납 편향을 네터워크에 넣음

### 2.3 Ray의 표기
다음 장의 이해를 위해 표기만 미리 언급하고 가자
$$r(t) = \vec o + t\vec d$$
- $o$ : 광선 시작점
- d : 픽셀 방향의 단위 벡터
- t : 광선을 따라가는 거리 파라미터

## 3. Volume Rendering Formulation
### 3.1 Ray(광선)와 Radiance Field의 역할
가상 카메라에서 한 픽셀을 찍는다는 건, 그 픽셀 방향으로 **광선(Ray)** 하나를 정면에 쏘는 것과 같다고 놓음.
(contiued...)