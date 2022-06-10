# Mask RCNN

# Mask RCNN

논문 : [https://arxiv.org/abs/1703.06870](https://arxiv.org/abs/1703.06870)

## 개요

Mask RCNN은 Faster RCNN 과 FCN의 결합한 구조의 모델이다. 

특징으로는 

- ROI - Align
- 기존 Bounding box regression과 Classification에 Binary Prediction추가
- 비교적 빠른 Detection시간과 높은 정확도
- 직관적이고 상대적으로 쉬운 구현

을 꼽을 수 있다. 

## Mask RCNN 구조

Mask RCNN의 구조를 살펴보기전에 Faster RCNN의 구조부터 살펴보자. 

![Screenshot from 2022-06-10 22-42-29.png](Mask%20RCNN%2037a3bf8fe93b4187838c871335e859a3/Screenshot_from_2022-06-10_22-42-29.png)

위 이미지의 형태가 Faster RCNN의 구조이다. 
먼저 Feature Extractor를 이용하여 Feature Map을 얻고 RPN(DNN를 이용한 Region Proposal)을 통해 ROI영역을 얻는다. 
얻은 ROI 영역에 ROI Pooling을 적용하여 Classification과 Regression을 수행하는것이 Faster RCNN의 구조이자 흐름이다. 

이 구조를 기반으로 ROI Pooling 이 ROI Align으로 바뀌고 Classification과 Regression Branch에 병렬적으로 FCN이 추가되면 Mask RCNN의 구조이며 아래 이미지와 같다. 

![Screenshot from 2022-06-10 22-45-51.png](Mask%20RCNN%2037a3bf8fe93b4187838c871335e859a3/Screenshot_from_2022-06-10_22-45-51.png)

이제 ROI Pooling을 왜 Align으로 바꿨는지, 또한 ROI Align이 뭔지에 대해서 알아보자. 

## ROI Align

### Segmentation에서 ROI Pooling의 문제점

![Screenshot from 2022-06-10 22-47-53.png](Mask%20RCNN%2037a3bf8fe93b4187838c871335e859a3/Screenshot_from_2022-06-10_22-47-53.png)

ROI Pooling은 Faster RCNN에서 RPN을 통해 나온 ROI영역을 FC Layer에 넣어주기위해 일정한 크기로 바꿔주는 역할을 한다. 

재각각의 형태의 ROI를 모두 같은 크기(7x7)로 변환해준다는 점은 효과적이었으나 이는 Object Detection에 제한적이었다. 

위 이미지를 참고하면 1차적으로 Feature Map에서 ROI 찾을때 소수점을 버리므로 위치정보가 손실되고 2차적으로 ROI Pooling을 실행할 때 손실된다. 

검출된 Object의 대략적인 위치만 찾으면 되는 Object Detection에 비해 Segmentation은 검출된 Object의 보다 구체적인 형태를 Pixel단위로 Classification해야하기 때문에 이러한 ROI Pooling의 구조가 위치정보를 상당히 손해보게 만들었다. 

이러한 문제점을 극복하기 위해 제안된 것이 ROI Align이다. 

### Bilinear Interpolation을 이용한 ROI Align

ROI Align은 위에서 언급한 ROI Pooling 진행시에 소수점 아래 정보가 버려지는 것을 방지하기 위해 Bilinear Interpolation을 사용한다. 

Bilinear Interpolation은 보간법의 일종으로 계산 방법은 다음과 같다. 

  

![Screenshot from 2022-06-10 23-12-27.png](Mask%20RCNN%2037a3bf8fe93b4187838c871335e859a3/Screenshot_from_2022-06-10_23-12-27.png)

채워야하는 곳으로 부터 알고있는 데이터들 사이의 거리를 고려하여 값을 채우는 방식으로 가로 세로의 차가 일정하게 하는 방식이다. 

이를 이용하여 ROI Align을 하는 방법은 

![Screenshot from 2022-06-10 23-47-00.png](Mask%20RCNN%2037a3bf8fe93b4187838c871335e859a3/Screenshot_from_2022-06-10_23-47-00.png)

예를 들어 2x2크기로 ROI Align을 한다고 가정하면 
ROI영역이 위 빨간 box의 형태로 나왔다. 이때 소수점 아래부분을 버리지 않고 그대로 image(Feature Map)상에 표시한다. 자연수의 형태가 아니기때문에 ROI 영역이 grid에 딱 맞지 않는다. 
이때 아래 순서를 따라 Bilinear Interpolation을 수행한다. 

1. ROI를 소수점 그대로 매핑하고 ROI의 개별 Grid에 4개의 Point를 균등하게 배열한다. 
2. 개별 Point에서 가장 가까운 Feature map grid를 고려하여 포인트부분을 Bilinear Interpolation(Weighted Sum)한다. 
3. 계산된 포인트를 기반으로 Max Pooling을 수행한다. 

위 방법을 이용하여 각각의 Grid의 Point를 계산하는 것이 위 이미지이다. 

위 결과로 나온 Point들에 대해서 Max Pooling을 해주기만 아래 이미지처럼 ROI Align을 해주게 되는 것이다. 

![Screenshot from 2022-06-10 23-52-33.png](Mask%20RCNN%2037a3bf8fe93b4187838c871335e859a3/Screenshot_from_2022-06-10_23-52-33.png)

이러한 방식을 통하여 ROI에 대한 손실된 위치정보(원래는 버려졌던 소수점 아래 숫자들)를 비교적 효과적으로 보존하며 가져갈 수 있기에 Segmentation을 수행할 경우 Object 형태에 대한 더욱 세밀한 표현이 가능했다. 

### Feature Extractor

![Screenshot from 2022-06-10 23-57-47.png](Mask%20RCNN%2037a3bf8fe93b4187838c871335e859a3/Screenshot_from_2022-06-10_23-57-47.png)

Mask RCNN의 경우 Feature Extractor 로 Resnet과 FPN이 결합된 형태의 구조를 사용한다. 

### Loss Function

- Mask RCNN Loss
$L = L_{CLS} + L_{bbox} + L_{mask}$
- Classification Loss
Muliclass cross entropy loss
- Bounding box Loss 
Smooth L1 loss
- Mask Loss 
k개의 정해진 Class에 대해서 그 Class의 pixel이 속하는지 그렇지 않은지 sigmoid로 결정 
(Binary cross entropy loss)

[https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fc0pdEg%2FbtqBL8vzmxg%2F1zkQAmbSKShCvdqXx8jXkk%2Fimg.png](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fc0pdEg%2FbtqBL8vzmxg%2F1zkQAmbSKShCvdqXx8jXkk%2Fimg.png)