---
layout: post
title:  "Density based Novelty Detection - 1"
date:   2018-11-05
desc: "Density based Novelty Detection - 1"
keywords: "Novelty Detection,Fault Detection,Outlier,Machine Learning, Gaussian,Mixture of Gaussian"
categories: [Machine learning]
tags: [Novelty Detection,Gaussian Estimation,MoG(Mixture of Gaussian)]
icon: icon-html
---

이번 포스팅에서는 Density based Novelty Detection 중에서 Gaussian Density Estimation과 Mixture of Gaussian Density Estimation에 대한 개념을 자세히 살펴보겠습니다. 그리고 이 자료는 고려대학교 강필성 교수님의 Business Analytics의 강의와 강의자료를 참고하였습니다.

---

## 1. Density-based Novelty Detection
>  
주어진 데이터를 기반으로 어떤 분포를 가지는지 밀도 함수를 구하는 것이 목적이라고 할 수 있습니다.  
아래 그림처럼 학습 데이터를 기반으로 만든 분포를 구하면, 새로운 데이터가 분포에서 낮은 확률로 나타나면 이상치로 인식하는 것이 밀도기반 이상치 탐지의 아이디어입니다.  

![](https://i.imgur.com/LsDoDAF.png?1)

---

## 2. Gaussian Density Estimation 
>  
가우시안 분포(정규 분포) 추정은, 학습하려고 하는 데이터의 분포를 정규 분포로 가정하는 것입니다.
기존 데이터에 가장 적합한 정규 분포를 찾고, 그를 기반으로 새로운 데이터가 어떤 확률을 가지는지를 통해 이상여부를 판단할 수 있습니다.

![](https://i.imgur.com/zgngdLR.png?1)  
위의 그림 처럼 데이터의 분포를 가우시안 분포로써 추정을할 수 있고, 새로운 데이터가 입력되었을때 그 데이터가 이 확률분포상에서 매우 낮은 값(구간의 95% 밖)을 가진다면 이상치로 판단할 수 있는 것입니다.

&nbsp;

#### 2-1. 가우시안 밀도 함수 추정 과정
 1. 먼저 아래 식과 같은 가우시안 분포를 가정한 것이기 때문에, 분포의 평균과 분산을 추정하면 됩니다.  
 
 $$p(x)= {1\over{(2\pi)^{d\over2} \left\vert \sum \right\vert^{1\over2} }} exp[{1\over2}(x-u)^T (\sum)^{-1}(x-u)]$$  
 
 $$u={ {1\over n} \sum_{x_i \in X^+} x_i} $$ (mean vector)  
 
 $$\sum = { {1 \over n} \sum_{x_i \in X^+} (x_i-u)(x_i-u)^T} $$ (covariance matrix)  
 
 여기서의 X+는 normal data에 해당하는 영역이라고 보면 됩니다.  
 
 2. 평균과 분산 추정시, 각 데이터들의 가우시안 분포 확률값의 곱이 최대가 되도록 평균과 분산을 잡아줍니다. 아래 그림을 보면 알 수 있듯이 특정 값에 해당하는 확률값들이 높은, 즉 왼쪽의 분포가 데이터의 분포를 좀 더 잘 설명한다고 할 수 있기 때문입니다.  
 
![](https://i.imgur.com/CEeEouY.png)  

 $$L=\prod_{i=1}^N P(x_i|u,\sigma^2) = \prod_{i=1}^N {1 \over {\sqrt{2\pi}\sigma} }exp(- { {(x_i-u)^2} \over {2\sigma^2}})$$  
 
 여기서 평균과 분산값은, normal data만을 이용해서 구한 값입니다.
 
 3. 위의 식을 log화 하고 미분하게 되면, 우리가 흔히 알고 있는 정규분포의 평균과 분산값을 얻을 수 있게 됩니다.  
 
 $$u={ {1\over N} \sum_{i=1}^N x_i} $$ (mean vector)  
 
 $$\sum = { {1 \over N} \sum_{i=1}^N (x_i-u)(x_i-u)^T} $$ (covariance matrix) 
 
&nbsp; 

#### 2-2. 공분산 행렬 형태에 따른 결과
위의 식에서 공분산 행렬로 Full 형태를 써야하지만, 연산 비용이 너무 크기 때문에 가능한 형태가 여러가지 존재합니다.  

 1. Spherical - 각 변수들의 분산합의 평균으로 모든 변수들의 분산값을 정의한 것으로, 항상 원 모양으로 그려집니다.  
 ![](https://i.imgur.com/kcSgTxR.png?1)  
 2. Diagonal - 변수간의 공분산은 0으로 정의하지만 변수들의 분산값을 다르게 지정하여 축방향은 유지하되 크기가 다르도록 만들어 주는 방법입니다. 연산 비용도 절약되면서 변수마다의 분산을 고려하기에 가장 많이 쓰인다고 합니다.  
 ![](https://i.imgur.com/AsLkFTJ.png?1)  
 3. Full - 원래의 공분산 행렬을 모두 가져다 쓴것으로 모양이 다양하게 나오지만, 변수의 수가 많아서 역행렬을 구하기 매우 어렵거나 불가능한 경우가 나올 수 있기 때문에 2번 방법을 많이 쓴다고 합니다.  
 ![](https://i.imgur.com/KdRWBKz.png?1)

&nbsp;

#### 2-3. 가우시안 분포 특징  

 - 공분산 행렬의 역행렬이 들어가기 때문에, 변수의 스케일에 민감하지 않다. 스케일링 불필요.
 
 - 이상치로 판단하는 최적의 영역을 확정할 수 있다.

&nbsp; 

---
## 3. Mixture of Gaussian Density Estimation
>  
위에서는 가우시안 분포를 하나만 가정했지만, 이번에는 여러개의 가우시안 분포가 있다고 가정하는 추정 방법입니다.
실제로 데이터는 넓은 범위에서 normal data들이 형성될 수 있기 때문에, 단일 가우시안보다 적합하다고 볼 수 있습니다.

![](https://i.imgur.com/QY5H8rl.png?1)

&nbsp;

#### 3-1. 혼합 가우시안 밀도 함수 추정 과정
 1. 이번에도 마찬가지로 여러 개의 가우시안 분포를 가정한 것이기 때문에, 여러 분포들의 평균과 분산들을 추정하면 됩니다.
 그런데 이번에는 한개의 분포가 아니기 때문에 아래의 식처럼 각 분포에 대한 가중치를 통해 선형결합으로 나타내면, 객체가 normal data에 포함될지에 대한 확률값을 얻을 수 있습니다.  
 
 $$p(x|\lambda) = \sum_{m=1}^M w_m g(x|u_m,\Sigma_m)$$  
 
 2. **Expectation-Maximization Algorithm**  

![](https://i.imgur.com/pRV8BNZ.png)  
 - Expectation  
 먼저 특정 객체가 주어졌을때, 가중치, 평균, 공분산을 이용해서 m개의 분포에 대해 각각 확률값을 구해줄 수 있습니다. 초기값은 임의로 설정하여 진행합니다.  
 
  $$p(m|x_i,\lambda) ={ w_mg(x_i|u_m,\Sigma_m) \over {\sum_{k=1}^M w_kg(x_i|u_k,\Sigma_k)} }$$  
  
 - Maximization  
 위에서 구한 객체들의 확률값을 분포마다 합산해줍니다. 그러면 분포들의 가중치를 새롭게 구할 수 있게 됩니다.  
 
 $$w_m^{(new)} = {1 \over N}\sum_{i=1}^N p(m|x_i,\lambda)$$  
 
 그리고 분포마다의 평균과 분산값도 객체들의 확률값을 이용해서 새롭게 구할 수 있습니다.  
 
 $$u_m^{(new)} = { {\sum_{i=1}^N p(m|x_i,\lambda)x_i}\over {\sum_{i=1}^N p(m|x_i,\lambda)} }$$    
 
 $$\sigma_m^{2(new)} = { {\sum_{i=1}^N p(m|x_i,\lambda)x_i^2}\over {\sum_{i=1}^N p(m|x_i,\lambda)} } - u_m^{2(new)}$$  
 
 
 
 3. 위의 과정의 계속해서 반복하다보면, 변화가 일어나지 않고 수렴될 것이고, 그 최종 결과의 분포들이 최적으로 선정되게 됩니다.  
 
 ![](https://i.imgur.com/l7RVls0.gif)
 
&nbsp;
 
#### 3-2. 공분산 행렬 형태에 따른 결과
 1. Spherical - 위와 마찬가지로 항상 원의 형태로 다양한 크기의 분포가 생기게 됩니다.  
 ![](https://i.imgur.com/Kaf63l8.png)  
 
 2. Diagonal - 축은 유지되면서 타원의 크기가 데이터에 맞게 조금 바뀌는 형태의 분포들로 이루어집니다. 가우시안 밀도 추정과 마찬가지로 가장 많이 쓰인다고 합니다.  
 ![](https://i.imgur.com/fdp1G9A.png)  
 
 3. Full - 데이터의 형태를 가장 정확하게 fit 시키지만 위와 마찬가지로 연산 비용이 많이 든다는 단점이 있습니다.  
 ![](https://i.imgur.com/ZTMqFcW.png)
 
 &nbsp;
 
#### 3-3. Code 구현

&nbsp;

- 먼저 다양한 가우시안 분포를 가지는 데이터를 형성하기 위해 반달 모양의 데이터를 만들었습니다.  

![](https://i.imgur.com/rkfv4kf.png)

- scikit-learn 패키지를 이용하여 위의 그래프를 먼저 두 개의 가우시안 분포를 가정해보았습니다.  

![](https://i.imgur.com/h9pUc4W.png)  

- 그랬더니 반달 모양의 데이터가 꼬아져있는 부분을 두 개의 가우시안 분포만으로는 설명할 수 없는 것이 보였습니다.  

- 위의 반달과 아래의 반달을 모양대로 분포를 만들기 위해서 가우시안 분포의 개수를 다양하게 변화시켜보았습니다.  

![](https://i.imgur.com/ozdYYLt.png?1)  

- 가우시안 분포 개수가 많아질수록 데이터의 형태대로 더 fitting됨을 알 수 있었습니다. 

- 저는 이 중에서 분포의 개수를 12개로 가정하고, 새로운 데이터가 들어왔을때 normal data에 속할지에 대한 test를 해보았습니다.

![](https://i.imgur.com/JWhi3mz.png)  

- 밑의 점의 경우 누가 보아도 모양체 안에 포함되서 그런지 normal data에 포함될 확률이 80% 이상이었습니다.  

- 이에 반해 위의 점의 경우는 매우 작은 확률을 가지기에 이상치로 판단하는데 문제가 없음을 알 수 있었습니다.

![](https://i.imgur.com/4eTwNzq.png)

&nbsp;

#### 3-4. 특징

 - 가우시안 추정의 확장판이라고 볼 수 있다.
 
 - 단일 정규 분포들의 선형결합으로 이루어진다.
 
 - 여러 정규 분포들의 조합이기 때문에, 편차가 더 적게 발생하지만, 여러 분포를 만들기 위해서는 훨씬 많은 데이터가 필요하다.

---

## 다음 포스팅에는..

계속해서 밀도 기반 추정 모델인, Parzen Window Density Estimation과 Local Outlier Factors에 대해서 살펴보도록 하겠습니다.

