# YOLO v3

# YOLO v3

## 개요

YOLO v3는 2017년 8월에 나온 Retinanet 이후인 2018년 4월에 나온 주목할 만한 Object Detection이다. 
주목할 점은 Retinanet의 특징인 FPN을 적극 도입함으로써 성능과 수행 속도를 향상시켰다는 것이다. 

v3가 v2에 비해서 달라진 점은 

- Feature Extractor를 Darknet 19 → Darknet 53으로 Upgrade
- Anchor box에 대해서 Ouput Feature Map당 3개, 서로 다른 크기와 스케일로 총 9개 사용한다.
- FPN사용

으로 볼 수있고, v2에서 주목할 만한 점이었던 K-Means Clustering은 v3에서도 사용한다. 

## YOLO v3 모델 아키텍처

![Screenshot from 2022-05-21 00-12-43.png](YOLO%20v3%2027f2d2a7f457495e84dc968be38cfcac/Screenshot_from_2022-05-21_00-12-43.png)

Yolo v3의 구조는 처음에 다소 복잡해 보이는 부분이 있다. 바로 FPN인데 알고보면 단순하다. 

먼저 Backbone모델을 통해 13 x 13의 형태까지 줄인다. 
13 x 13 Feature Map에 Conv 를 통해 나온 Feature 가 위 이미지의 P_5이다. 
그리고 13x13 Feature Map과 26 x 26의 형태의 Feature Map과 합치는데 13x13을 Upsample하여 크기를 키운다. 
이 Feature Map에도 Conv를 통해 P_4를 만든다. 
마지막으로 26 x 26 Feature Map을 Upsample 하여 52x52 Feature Map과 합치고 이에 Conv를 하여 P_3를 만든다. 

이렇게 나온 3개의 Feature Map을 통하여 나머지 작업(Head)을 진행하게 된다. 

FPN은 CNN의 특성을 이용했다고 볼 수 있는데
깊이가 깊어질수록 세세한 표현에 대한 정보는 줄어들고 대략적인 표현을 잘 기억한다. 
하지만 우리는 두 특징을 모두 보존한 형태로 Object Detection을 진행하고 싶기 때문에 각각의 특징을 잘 살릴수 있는 Feature Map들을 합치는 방식을 이용한 것이다. 
결국에는 이 방법이 이미지의 특징을 더욱 잘 보존한 채로 패턴을 잘 파악할 수 있게 해준다. 

아래 이미지를 참고하면 더욱 이해하기가 쉽다. 

![https://miro.medium.com/max/1400/1*d4Eg17IVJ0L41e7CTWLLSg.png](https://miro.medium.com/max/1400/1*d4Eg17IVJ0L41e7CTWLLSg.png)

## Output Feature Map

![https://blog.paperspace.com/content/images/2018/04/yolo-5.png](https://blog.paperspace.com/content/images/2018/04/yolo-5.png)

위 FPN을 통해 3개의 Feature Map에서 정보를 추출해 낸다고 했는데 그 형태는 위 이미지와 같다. 

v3는 각각의 Cell을 기준으로 3개의 Anchor box를 가지고 있다. (v2에서는 5개였음)

각각의 셀은 3개의 Anchor box에 대한 정보를 담고 있는데 

**[ 좌표(2) + 높이너비(2) + Object Confidence(1) + Class Scores(20)]**
의 구조로 1개의 Anchor box당(Pascal VOC기준) 25개의 정보를 가지고 있으므로 
1개의 Cell 당 총 75의 Depth를 가지고 있다.

최종적으로 3개의 Feature Map에서 뽑아내므로  
**13x13x75 , 26x26x75 , 52x52x75 크기의 Feature Map**들이 나온다는 것을 알 수 있다.