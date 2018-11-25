---
layout: post
title:  "Concept of Novelty Detection"
date:   2018-10-30
desc: "Concept of Novelty Detection"
keywords: "Novelty Detection,Fault Detection,Outlier,Machine Learning"
categories: [Machine learning]
tags: [Novelty Detection,Outlier]
icon: icon-html
---

이번 포스팅에서는 많은 분야에서 널리 활용되고 있고 연구되고 있는, 이상치 탐지(Novelty Detection)에 대한 개념을 자세히 살펴보겠습니다. 그리고 이 자료는 고려대학교 강필성 교수님의 Business Analytics의 강의와 강의자료를 참고하였습니다.

---

## 1. What is Novel Data?
>  
Novel Data, Outliers. 즉, 다른 데이터들에 비해 다른 성질을 가지는 데이터.  
 (다른 데이터들과는 다른 매커니즘에 의해 형성된 데이터, 다른 분포를 가지는 객체들이라고 볼 수 있습니다.)  
 
 &nbsp;
 
### 1-1. Type of Novel Data

&nbsp;

1. **Global Outlier**  
아래 그림과 같이 다른 데이터들과는 확실하게, 상당하게 동떨어져 있는 데이터를 의미합니다.  
![](https://i.imgur.com/6X37rai.png?1)

2. **Contextual Outlier**  
아래의 그림은 연간 온도의 변화량을 나타낸 그래프입니다. Normal로 표시된 부분에서의 온도와 Anomaly 부분에서의 온도는 비슷한 값을 가지지만, 앞뒤의 온도 변화 맥락을 고려하면 왜 같은 값인데도 Novel data로 판별되는지 알 수 있습니다.  
![](https://i.imgur.com/wejkB9D.png)

3. **Collective Outlier**  
아래 그림을 보면, 검은 객체 하나하나가 전체 데이터 측면에서는 이상치가 아니지만, 검은 원들이 모이는 군집은 전체 원들의 배치와는 다른 형태를 띄는 것을 볼 수 있습니다. 이런 형태의 데이터들을 Collective Outlier라고 정의합니다.  
![](https://i.imgur.com/XSGUXB5.png)

&nbsp;

### 1-2. Noise data?
 - Noise는 무작위하게 발생되는 에러로써, Control이 가능합니다.
 
 - 회귀식을 예로 들면, 각 data point들은 모두 회귀식에 의해 100% 설명되지 못하는데, 여기서 발생되는 차이, 즉 설명되지 않는 부분들을 noise라고 볼 수 있습니다.  
 
 - 과거에는 Noise data가 모델의 성능을 방해한다고 여겨서 이를 제거하는 다양한 방법들을 써왔습니다.  
 
 - 최근에는 GAN과 같은 방법론에서 좀 더 좋은 성능을 내기 위해 일부러 noise를 생성하여 활용하는 현상도 일어나고 있습니다. (원래의 이미지 데이터에서 Noise를 추가하면서 같은 label의 비슷한 이미지들을 생성할 수 있음)

&nbsp;

## 2. Novelty Detection
>  
Anomaly Detection, Fault Detection으로도 불리며, normal data를 형성하는 매커니즘을 방해하는 이상치를 탐지하는 기법
비지도 학습으로, 다른 객체들에 비해 다른 성질(거리, 분포, 밀도 등)을 가지는 특정한 객체를 판별해내는 기법

정상인지 아닌지를 판별한다는 개념에서 Classification과 유사한점이 많으므로, 둘을 비교하면서 내용을 진행하겠습니다.

&nbsp;

### 2-1. Classification vs Novelty Detection
![](https://i.imgur.com/NTdCtTW.png)  
왼쪽 그림은 두 범주에 대한 Classification으로, 새로운 데이터 A와 B에 대해서 O 범주로 분류하게 됩니다.
그러나 오른쪽 그림에서는 네모칸이 쳐진 O 부분에 들어오지 않는 데이터는 이상치라고 판단하게 됩니다. 
-> 두 알고리즘 모두 A와 B에 대해서 class를 판별한다는 공통점이 있지만, 실제로 둘은 매우 다른 특징을 가지고 있습니다.  

- 먼저, Classification은 모든 범주의 데이터를 학습에 이용해서, 새로운 데이터가 주어졌을때 어떤 범주에 속하는지 판별을 해주는 모델입니다. 그만큼 모든 범주에 대해서 충분한 데이터로 학습되어야 한다는 의미입니다.  

- 그러나 Novelty Detection은 Normal data, 즉 이상치로 labeling되지 않은 데이터들만을 학습에 이용하고, 새로운 데이터가 그 영역안에 포함되지 않은 경우에만 Novel Data로 판별해주는 모델입니다. 특정 label의 데이터 수가 극히 적을 경우 분류학습을 하기에는 성능이 매우 떨어지기 때문에 이럴경우 이상치 탐지 기법을 사용하는 것입니다. 

![](https://i.imgur.com/P5qTpIP.png?1)  
위 그림과 같은 데이터에서는 붉은색 label에 대한 객체가 적기 때문에, 이를 충분히 학습하지 못할 것이고 새로운 객체에 대해서 정확하게 분류를 해낼수가 없게됩니다. 따라서 normal data들을 충분히 학습해서 일반적인 데이터의 분포를 확실하게 학습하게 하고 그에 벗어나는 것을 탐지해내고자 하는 것이 Novelty Detection의 아이디어 입니다.

&nbsp;

### 2-2. Novelty Detection의 문제  

&nbsp;

1. labeling이 잘 된 경우 학습 데이터의 기준이 명확하지만, 그렇지 않은 경우 normal data의 범위를 잡는 것이 주관적일 수 있습니다.   

2. 1번 문제의 연장선으로, 같은 scale의 데이터라도 어떤 도메인의 데이터냐에 따라 이상치의 기준이 달라질 수 있습니다. 예를 들어, 같은 온도에 대한 데이터라도 일반 공장에서 발생한 기기의 온도인지, 아이스크림 공장에서 발생한 제품 생성 기기의 온도인지..(너무 극단적이긴 하지만)에 따라 예민하게 이상치를 잡아내야하는지 기준이 다른 경우가 있을 수 있습니다.

3. 거리나 밀도를 기반으로 이상치를 탐지하지만, 정확히 어떤 요인때문에 그렇게 판단했는지는 설명하기 힘듭니다. 다양한 변수들을 기반으로 거리가 동떨어져서 이상치로 탐지했다 하더라도, 정확히 어떤 변수가 얼만큼 이상한지에 대해 설명하기가 힘들다고 합니다.(그러나 이상치 탐지와 함께, 이상 요인을 분석하는 기법들도 연구되고 있습니다. 추후에 포스팅하겠습니다.)

&nbsp;

### 2-3. 성능 평가 기준

&nbsp;

##### 1. Confusion Matrix
아래 그림과 같이 무엇으로 예측했을 때 실제값이 어떤지, 에 대한 값을 표로 나타낼 수 있습니다.  
![](https://i.imgur.com/AbCTO5z.png)  
그러면 이 값들을 이용해서 얼마나 Novel data를 잘 잡아내는지, 예측을 얼마나 잘못하는지에 대한 지표를 아래의 표처럼 정의할 수 있습니다.  
![](https://i.imgur.com/VieVIoM.png)

 위의 A,B,C,D 값은 객체 하나하나에 대해서 포함되는 부분에 1씩 더해주는 방식입니다. 
 즉, 특정 객체가 Normal 데이터에서 조금 벗어난 Novel 객체와, 심각하게 멀리 떨어진 Novel 객체를 같게 생각한다는 의미입니다. 
 이렇게 되면 객체들을 너무 strict하게 판단하므로 결과를 판단하는 기준으로는 대표성이 조금 떨어지는 것처럼 보입니다.

&nbsp;

##### 2. Equal error rate
아래의 그림은 위에서 구한 FRR과 FAR를 그래프로 그린 것입니다.  
![](https://i.imgur.com/QCEE1Pi.png)  
EER은 FAR과 FRR이 같아지는 지점으로, 이상치 탐지 알고리즘 성능 평가 기준으로 사용할 수 있습니다.  
위의 그림에서 검은색영역이 작아질수록, 당연스럽게도 데이터를 잘못 판별하는 두 지표가 줄수록 좋은 알고리즘이라고 볼 수 있습니다.

EER 이 지표가 Confusion Matrix에 비해서 좀 더 유연한 성능 평가 기준이라고 생각합니다.  

&nbsp;

---

## 3. Application Field

&nbsp;

- 공정 : 설비의 상태를 나타내는 신호 데이터나 공정의 진행 상황에 대한 데이터가 실시간으로 들어올 때, 일반적인 상황과 다른 데이터가 발생할 경우 공정에 문제가 생겼음을 빠르게 진단하여야하기 때문에, 이상치를 정확하고 빠르게 진단하는 기법이 필요합니다.

- 카드사용 : 카드는 매우 중요한 금전적인 부분인데, 만약 카드를 사용한 장소나 금액이 평소의 패턴과 많이 벗어난다면 빠르게 판단하여 도용이나 도난을 의심해보아야 합니다.

- 계정 : 이 부분은 제가 직접 당한 것으로, 네이버 아이디가 갑자기 로그인이 정지되어서 확인해보니, 갑자기 중국에서 로그인된 이력이 생겼다고 합니다. 이처럼 평소와는 매우 다른 패턴의 기록을 통해 이상치로 판단하는 과정이 중요함을 알 수 있었습니다.

---

## 다음 포스팅에는..

Novelty Detection 에는 Density-based(밀도기반), Distance/Reconstruction-based(거리기반), Model-based 등 다양한 방법론들이 있습니다. 다음 포스팅에서는 Density-based Novelty Detection 중에서 Gaussian Density Estimation과 Mixture of Gaussian Density Estimation에 대해 먼저 살펴보겠습니다.
