---
layout: post
title:  "Density based Novelty Detection - 2"
date:   2018-11-13
desc: "Density based Novelty Detection - 2"
keywords: "Novelty Detection,Fault Detection,Outlier,Machine Learning, Parzen Window,LOF(Local Outlier Factors)"
categories: [Machine learning]
tags: [Novelty Detection,Parzen Window, LOF(Local Outlier Factors)]
icon: icon-html
---


이번 포스팅에서는 Density based Novelty Detection 중에서 Parzen Window Density Estimation과 Local Outlier Factors에 대한 개념을 자세히 살펴보겠습니다. 그리고 이 자료는 고려대학교 강필성 교수님의 Business Analytics의 강의와 강의자료를 참고하였습니다.

---

## 1. Kernel-density Estimation
>  
앞선 포스팅에서 소개한 가우시안 밀도 추정과는 다르게, 어떠한 분포도 가정하지 않고 데이터로부터 분포를 매우 불규칙하게 추정해내는 방법입니다. 따라서 아래 그림처럼 데이터 자체가 분포가 됨을 볼 수 있습니다.  

![](https://i.imgur.com/M96i0bN.png)  

1. x에 대한 분포가 p(x)로 나타내어질 때, 특정 x가 R이라는 공간에 포함될 확률은 아래와 같이 나타낼 수 있습니다.  

 $$P = \int_R p(x) dx$$  
 
2. N개의 x가 주어졌다고 가정했을때, N개 중 k개가 위에서 말한 R이라는 공간안에 포함될 확률은 아래처럼 나타낼 수 있습니다.(포함되거나 포함되지 않거나인 이항 분포 이므로)  

 $$P(k) = \binom{N}{k}P^K(1-P)^{N-k}$$  
 
3. 위의 식으로부터 k/N의 기댓값과 분산을 아래와 같이 구할 수 있습니다.  

 $$E[{k \over N}]=P$$  
 
 $$V[{k \over N}]={P(1-P) \over N}$$

4. 여기서 N이 무한대로 커지게 되면 대부분의 데이터가 평균지점에 몰리게 됩니다. 따라서 이 경우 P값 자체를 k/N으로 정의해버릴 수 있습니다.  

 $$P={k \over N}$$ 
 
5. 그리고 위에서 정의한 구간 R이 매우 작다면, 직선처럼 여겨지므로 아래와 같이 재정의할 수 있습니다.  

 $$P = p(x)V$$
 
6. 위의 두 식을 이용하면 최종적으로 x의 분포 p(x)를 아래와 같이 정리할 수 있습니다.  

 $$p(x) = {k \over {NV}}$$
 
 - 여기서 샘플 수인 N이 커지고, 포함 구간인 V(volume)값이 작아지면 좀 더 정교한 추정이 될 수 있습니다.
 
 - 그리고 어느정도의 샘플을 포함하면서, 확률값이 일정하도록 유지시키는 적당한 V값을 찾는 과정이 중요합니다.
 
 - k값을 고정시키고 V값의 변화를 찾는다면 knn 알고리즘(뒤에서 설명할 예정)이 될 것입니다.
 
 - V값을 고정시키고 k값의 변화를 찾게 되면 그것이 **Kernel-density estimation** 입니다.
 
## 2. Parzen Window Density Estimation 
>  
창문처럼 각진 다면체를 특정 포인트를 중심으로 만들고, 그 다면체 안에 들어오는 객체의 수를 세는 방식입니다. 
그러나 들어오거나 아니거나 하는 극단적인 카운트 방법은 비연속적이고, 거리를 전혀 고려하지 못하기에 거리에 따른 확률값으로 대체하는 커널을 이용할 수 있습니다.

![](https://i.imgur.com/AaVF1CT.png)

- 아래의 왼쪽 그림처럼 특정 포인트를 기준으로 사각 모양안에 들어오면 1값을 부여하지만 아니면 모두 0으로 부여합니다. 그래서 오른쪽과 같은 형태의 다양한 커널들이 이용될 수 있습니다.

![](https://i.imgur.com/EvACoyM.png)

- 아래는 주변의 거리를 어느정도까지 볼 것인지 h값 변화에 따른 분포 형성입니다. h 값이 클수록 큰 범위에 있는 점들을 고려하기 때문에 전체적으로 smooth 해지는 것을 알 수 있고, 작아질수록 많은 분포들이 생기는 것을 볼 수 있습니다.

![](https://i.imgur.com/848bFJB.png)  

- 아래 그림을 보면 가우시안 추정이나 혼합 가우시안 추정은 일관된 모양으로 분포가 이루어지는 것을 알 수 있지만, Parzen window의 경우 각 점들을 기준으로 가우시안 커널을 기반으로 한 값들이 분포화되기 때문에 다양한 모양을 띄는 것을 알 수 있습니다.
![](https://i.imgur.com/5JuggeP.png)


## 3. Local Outlier Factors(LOF)
>  
전체의 데이터 측면에서는 아니지만, 지역적으로는 이상치가 될 수 있는 점들을 찾아내는 기법입니다.
아래 그림처럼 O1은 Global한 측면에서 이상치이지만, O2는 C2 군집의 관점에서 봐야 이상치로 찾아낼 수 있습니다.

![](https://i.imgur.com/HxneJhK.png)

#### 3-1. LOF 값 추정 과정
 
 1. **k-distance(p)** 정의
  - 객체 p를 기준으로 k번째로 가까이 있는 점까지의 거리
 2. **N(p)** 정의
  - 1에서 정의한 k-distance(p)보다 같거나 가깝게 있는 점들의 수
 3. **reachability distance** 정의
  - k-distance보다 작거나 같으면 k-distance값을 쓰고, 크면 실제 거리값을 쓰는 개념  
  
 ![](https://i.imgur.com/FupmsPw.png?1)
 
 4. **local reachability density** 정의
  - p의 밀도를 구할때, k-distance(p) 안에 있는 점들과 p 사이의 거리를 3번에서의 reachability distance로(해당 점을 기준으로 구해야 함!) 구해서 모두 더해줍니다. 그리고 이 값으로 그 점들의 개수를 나눠줌으로써, 어느 정도의 밀도를 가지는지 정의해줄 수 있습니다.
 $$lrd_k(p) ={ |N_k(p)| \over {\sum_{o \in N_k(p)} reachability-distance_k(p,o)} }$$  
 
 5. **local outlier factor*** 정의
  - 기준으로 하는 p점의 밀도뿐 아니라, k-distance(p)에 들어오는 점들의 밀도를 모두 고려해서 상대적인 밀도값을 산출해낼 수 있기 때문에 다른 점들과 얼마나 떨어져있는지 판단할 수 있는 척도가 됩니다.

$$LOF_k(p)={ { {1 \over lrd_k(p)}\sum_{O \in N_k(p)} lrd_k(o) } \over |N_k(p|}$$
![](https://i.imgur.com/WolukLz.png)

- Case1과 Case3에서는 파란점의 lrd(4번 정의 참고)와 주위 녹색점들의 lrd값이 유사해서, 즉 주변 밀도가 비슷하므로 이상치로 판단되지 않습니다.
- 그러나 Case2에서는 파란점의 lrd는 낮지만, 녹색점들의 lrd는 높기 때문에 파란점을 이상치로 볼 수 있게 됩니다.


#### 3-3. Code 구현 
![](https://i.imgur.com/MFAzNhE.png)  
(참고: scikit-learn.org)
![](https://i.imgur.com/VUeFO6w.png)  

- 위의 그림에서 각 점들의 Local Outlier Factor 값의 크기를 붉은 원의 크기로 표현했습니다.
- 노란별로 표시된 두 점을 보면 왼쪽 아래 점을 오른쪽 아래의 점처럼 어느정도 이상치로 판단하는 것을 알 수 있습니다.
- 왼쪽 아래 점이 정상 데이터 같지만 LOF측면에서 이상치로 골라내고 있다는 점을 보여주고 있습니다.



#### 3-4. 특징
 1. 전체의 구조에서는 알 수 없는 이상치를, 데이터의 지역적인 특성에 맞춰서 찾아낼 수 있다는 것이 장점입니다.  
 2. 그러나 LOF값이 어느정도 이상이어야 이상치로 판단하는지에 대한 기준을 정하는 것이 상당히 주관적일 수 있습니다.

## 다음 포스팅에는..

모델을 기반으로 이상치를 탐지하는, 1-SVM과 SVDD(Support Vector Data Description), Isolation Forest에 대해서 살펴보도록 하겠습니다.
