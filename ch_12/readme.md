# EfficientDet

Date: 2022년 5월 30일 → 2022년 6월 3일
chapter: 12

논문 링크 : [https://arxiv.org/abs/1911.09070](https://arxiv.org/abs/1911.09070)

# EfficientDet

## 개요

![Untitled](EfficientDet%20686af061930542d5b0f7823584b525c5/Untitled.png)

EfficientDet은 Classification 모델인 EfficientNet모델을 Backbone으로 사용하고 기존의 FPN을 개량한 형태인 BiFPN을 Neck으로 사용한 Object Detection모델이다. 

EfficientDet의 핵심은 크게 두 가지로 볼 수 있다. 

1. BiFPN 
2. Compound Scaling 

BiFPN은 위 이미지에서 보이듯이 Feature map을 복잡하게 결합하여(Feature Fusion) 최적의 결과를 내도록하는 방법이다. 
Compound Scaling은 backbone인 EfficientNet에 도입된 개념으로 네트워크의 깊이와 너비와 해상도를 최적으로 조합하여 모델의 성능을 극대화 시키는 방법으로 Object Detection에서도 이 방법을 도입하였다. 

![Screenshot from 2022-06-07 21-20-23.png](EfficientDet%20686af061930542d5b0f7823584b525c5/Screenshot_from_2022-06-07_21-20-23.png)

EfficientDet은 기존 모델들에대해 비교적 적은 연수 수와 파라미터 수를 가지고 타 모델보다 높은 모델 예측 성능을 나타냈다. 

## BiFPN (Bi directional FPN)

![Screenshot from 2022-06-07 21-22-24.png](EfficientDet%20686af061930542d5b0f7823584b525c5/Screenshot_from_2022-06-07_21-22-24.png)

EfficientDet의 핵심적인 기능 중 하나인 BiFPN이다. 

BiFPN은 두 가지 핵심적인 내용을 가지고 있다. 

1. Cross Scale Connections 
2. Weighted Feature Fusion 

**Cross Scale Connections**은 모델의 Neck에 해당하는 FPN의 구조를 복합적으로 설계해 가장 최적의 성능을 나타내는 구조를 뜻한다. 
FPN에서 구현하고자 하는 바는 다양한 형태의 Feature map을 융합하여 Feature Fusion을 시도하는 것이다. 

위 이미지에서 볼 수 있듯이 단순한 형태의 FPN부터 
거꾸로 올라가며 다시 더하는 형태의 PANet과 
NAS 를 통해 얻은 NAS-FPN과 
최종적으로 EfficientDet에 채택된 형태인 BiFPN이 존재한다. 

**Weighted Feature Fusion**은 FPN구조에서 단순히 Feature map을 더하는 형태가 아니라 더하는 과정에 각각의 Feature map에 가중치를 추가시켜 더하는 방법이다. 이 가중치는 학습되는 값이다. 

### BiFPN - Cross Scale Connection

![Screenshot from 2022-06-07 21-37-03.png](EfficientDet%20686af061930542d5b0f7823584b525c5/Screenshot_from_2022-06-07_21-37-03.png)

초기의 FPN은 Feature map의 크기가 작아질수록(깊어질수록) 큰 object를 잘 찾고(패턴) 이러한 특징을 보존한채 이전 단계의 Feature map에서 찾기위해 Upscaling을 한 후 더해주는 구조이다. 

이러한 구조를 이용하여 성능 향상에 큰 도움이 되었고 이후로 FPN에 대한 연구가 활발히 진행되었다. 

이후로 나온 것이 PANet이며 초기의 FPN에 대한 결과를 다시 downsampling하며 더해가는 구조이다.
FPN이 high level의 Feature 를 low level에 더해줬다면 PANet은 추가적으로 low level Feature 를 high level에 더해주는 것이다. 

NAS-FPN은 NAS Search를 통해 얻어낸 구조이다. 굉장히 많은 시간이 걸렸으며 직관적으로 효과적인 이유를 알수는 없지만 성능은 향상되었다고 한다. 

BiFPN은 최종적으로 EfficientDet에 적용된 방법으로 복합적이지만 규칙적인 구조를 가지고 있다. 
PANet을 개량한 버전이라고 볼 수 있다. 핵심적인 것은 이러한 형태를 하나의 block으로 여러번 진행한다는 것(repeated blocks)이다.

### BiFPN - Weighted Feature Fusion

![Screenshot from 2022-06-07 21-43-51.png](EfficientDet%20686af061930542d5b0f7823584b525c5/Screenshot_from_2022-06-07_21-43-51.png)

Weighted Feature Fusion은  서로 다른 resolution(feature map size)를 가지는 input feature map들을 Output feature map을 생성하는 기여도가 다르기 때문에 서로 다른 가중치(학습하여 얻은)를 부여하여 합치는 방법이다. 

위 이미지에서 보이는 w가 가중치로 정규화 과정까지 포함하여 계산한다. 

![Screenshot from 2022-06-08 16-21-11.png](EfficientDet%20686af061930542d5b0f7823584b525c5/Screenshot_from_2022-06-08_16-21-11.png)

최종적으로 BiFPN을 적용하여 위 이미지와 같은 성능을 이끌어냈는데 
Parameters가 줄어든 이유는 BiFPN Network구현 시 Separable Convolution을 적용하였기 때문 

## Compound Scaling

Compound Scaling은 네트웍의 깊이, 필터 수 , 이미지 resolurution크기를 **함께 최적으로 조합하여 모델 성능을 극대화하는 방법**이다. 

EfficientDet에 적용되기 이전에 Backbone인 EfficientNet에 적용이 되었으며 Compound Scaling에 대해서 EfficientNet을 통해 알아보도록 한다. 

### EfficientNet

![Screenshot from 2022-06-08 16-31-20.png](EfficientDet%20686af061930542d5b0f7823584b525c5/Screenshot_from_2022-06-08_16-31-20.png)

EfficienNet은 위 이미지처럼 모델의 전체 구조를 튜닝하는 것이 핵심적인 모델이다. 

필터 수, 네트웍의 깊이, resolution 의 세 가지 분야에서 튜닝(최적화)을 시도하는데 이를 각각 최적화하는 것이 아니라 이들이 어떻게 튜닝이 되어야 가장 좋은 성능을 내는 조합이 되는지를 찾는것이 바로 Compund Scaling이다. 

![Screenshot from 2022-06-08 16-31-40.png](EfficientDet%20686af061930542d5b0f7823584b525c5/Screenshot_from_2022-06-08_16-31-40.png)

위 세 가지 종류에 대해서 튜닝을 시도할수록 어느정도 수준이 되면 성능 향상이 줄어든다. 

따라서 각각의 성능향상보다는 세 종류에 대한 조합을 찾을 필요성을 생김. 

![Screenshot from 2022-06-08 16-31-52.png](EfficientDet%20686af061930542d5b0f7823584b525c5/Screenshot_from_2022-06-08_16-31-52.png)

 

EfficientNet에서 Compound Scaling을 진행한 방식은 다음과 같다. 

1. 최초에 $\phi$  는 1로 고정시키고 Grid Search(Hyper  Parameter Tuning)를 이용하여 최적의 $\alpha, \beta, \gamma$ 를 찾는다. 이때의 모델이 B0모델이다.
2. 이후 $\phi$  를 증가시키며 B1 ~ B7까지 Scale Up한다. 

연산량이 지나치게 커지는 것을 방지하기위해  $\alpha, \beta, \gamma$ 에 대한 제한을 둔다. 

$\alpha$ 는 Depth를 의미하며 늘어날수록 연산량은 비례하여 늘어나지만 나머지 값들은 4배가 늘어나므로 해당 계수에 제곱한 값을 적용하여 제한한다.

![Screenshot from 2022-06-08 16-32-16.png](EfficientDet%20686af061930542d5b0f7823584b525c5/Screenshot_from_2022-06-08_16-32-16.png)

위 과정을 통해 탄생한 EfficientNet의 Hyper Parameter 에 대한 값은 위와같으며 B7이 가장 좋은 성능을 낸다. 

![Screenshot from 2022-06-08 16-32-28.png](EfficientDet%20686af061930542d5b0f7823584b525c5/Screenshot_from_2022-06-08_16-32-28.png)

B0부터 B7까지의 성능은 위 이미지의 그래프와 같으며 속도와 정확도가 반비례하므로 용도에 맞게 사용하여야 한다. 

### EfficientDet Compound Scaling

너무 거대한 Backbone, 여러 겹의 FPN, 큰 입력 이미지의 크기 등의 개별적인 부분들에 집중하는 것은 비효율적이며 EfficientNet에서 여러 요소를 함께 Scaling하며 최적의 결과를 얻은것과 같이 EfficientDet에서도 Backbone, FiFPN, Prediction layer, 입력이미지의 크기에 Compound Scaling을 적용하여 D0 ~ D7까지의 최적의 모델을(조합을) 찾아낸다. 

 EfficientDet에서의 Compound Scaling은 다음과 같은 항목에서 이루어진다. 

1. Backbone
2. BiFPN - Channels/ layers
3. Prediction Network
4. Input Image Resolution 

이렇게 5개의 항목에 대해서 최적의 조합을 찾아낸 것이 EfficientDet이다. 

**< Backbone>** 

EfficientNet B0 ~ B7로 Scaling 그대로 적용 한다. 
즉, D0는 B0를 사용하고 D1은 B1을 사용한다. 
다만, D7의 경우에는 B6를 사용하고 D7x에 B7을 사용한다. 

**<BiFPN>**

BiFPN에 대한 항목은 2가지로 나뉜다. 

Depth(layers)는 BiFPN의 기본 반복 block을 3개로 설정하고 Scaling을 적용하며 동일한 Block이 몇 번 반복하는지를 결정한다. 

$D_{bifpn} = 3 + \phi$

Width(channels)는 Feature map에 1x1 conv를 진행해 줄때의 filter의 Channe 크기를 의미한다. 
[ 1.2, 1.25, 1.3, 1.35, 1.4, 1.45 ]중에서 Grid Search를 진행하여 얻은 값(1.35 당첨)을 계수로 사용하여 적용한다. 

$W_{bifpn} = 64 * (1.35^{\phi})$

**<Prediction Network>**

Class Prediction Network와 Box Prediction Network에서의 block의 갯수를 몇개로 설정할지를 결정한다. BiFPN과 동일한 개념으로 아래 식을 적용하여 사용한다. 

$D_{box} = D_{class} = 3 + [\phi/3]$

**<입력 이미지 크기>**

기본적인 입력 이미지의 크기를 512로 하고 $\phi$ 의 크기에 따라 늘려간다. 

$R_{input} = 512 + \phi*128$ 

## 기타적용 요소 및 성능 평가

- actibation : SiLU(SWISH)
- Loss : Focal Loss
- Augmentation : horizontal flip, scale jittering
- NMS : Soft NMS

![Screenshot from 2022-06-08 18-56-03.png](EfficientDet%20686af061930542d5b0f7823584b525c5/Screenshot_from_2022-06-08_18-56-03.png)

다른 모델에 비해 연산량이 현저히 적고 성능또한 뛰어난 것을 확인할 수 있다.