# RetinaNet

Date: 2022년 5월 30일 → 2022년 6월 3일

# RetinaNet

![Untitled](RetinaNet%2066449fc824334333ac7ffec737b39d64/Untitled.png)

RetinaNet은 One-stage Detector에서 Yolo v2가 나온 이후에, Two-stage Detector로는 Faster-RCNN이 나온 이후에(가장 주목받고 있을 시기) 나온 One-stage Detector이다.

수행속도 측면에서 YOLO 나 SSD보다 느리지만 Faster-RCNN보다 빨랐으며, 성능면에서는 타 Detection모델보다 좋았으며, 특히 작은 Object에 대해서 Detection성능이 좋았다.

**<특징>**

1. Focal Loss
2. Feature Pyrarmid Network

## 1. Focal Loss

### < Cross Entropy의 문제 >

Focal Loss는 Cross Entropy의 변형형태라고 볼 수 있다.

먼저 cross entropy는 다음과 같다.

![Untitled](RetinaNet%2066449fc824334333ac7ffec737b39d64/Untitled%201.png)

![https://s26.postimg.cc/oygxf6kll/2018-04-16_11.01.43.png](https://s26.postimg.cc/oygxf6kll/2018-04-16_11.01.43.png)

Cross Entropy는 -ln 함수를 이용한 Loss Function이며 GT가 1일 경우 확률이 높으면 Loss값이 작아지고
GT가 0일 경우 확률이 높으면 Loss값이 커진다.

이러한 CE의 문제점은 다음과 같은 상황에서 발생한다.

forground(easy example)에 대해서 0.9의 확률로 찾아낸다면 loss는 0.1053이다.
forground(hard example)에 대해서 0.1의 확률로 찾아낸다면 loss는 2.3025이다.

위 두 가지 상황을 비교해 보면 loss값이 바람직하게 나오는 것을 확인할 수 있지만 easy example은 데이터셋의 특성상 hard example보다 압도적으로 수가 많다. 따라서 easy example에 대한 loss의 총합이 hard example의 loss의 총합을 압도해 버려 모델이 hard example에 대해서 학습하기 보다는 easy example에 대해 학습한다.

### **< One-Stage Object Detector의 Class Imbalance문제 >**

![Untitled](RetinaNet%2066449fc824334333ac7ffec737b39d64/Untitled%202.png)

train image의 대부분이 Object보단 background 영역이 대부분이다.  
이러한 Background에 대한 Loss 값은 크지 않지만 그 수가 굉장히 많기 때문에(One-Stage Detector의 특징인 Anchor box를 사용하기 때문) 객체에 대한 Loss값보다 더욱 도드라지게 된다.

즉, 찾기 어려운 object들을 더 잘찾기위한 성능개선 보다는 이미 잘 찾는 object들에 대해서 더 잘 찾으려는 성능개선의 방향으로 학습이 진행된다.

<aside>
🔥 Two-Stage Detector의 경우 Region Proposal을 사용하기 때문에 이미지의 배경은 무시하고 객체로 추정되는 영역만 보고 학습을 하기 때문에 이러한 문제에 영향을 덜 받는다. ( Faster-RCNN이 정확도가 높은 이유)

하지만 One-Stage Detector의 경우 Region Proposal없이 모든 영역을 보고 Anchor box를 이용해 학습을 하기때문에 Easy Negative같은 학습에 대한 방해요소에 대해 고려해야 한다.

</aside>

따라서 이 문제를 해결하기 위해 Easy example에 대한 Loss값을 조절하여야 하고
그 방안으로 나온 것이 Focal Loss이다.

### < Focal Loss >

![Untitled](RetinaNet%2066449fc824334333ac7ffec737b39d64/Untitled%203.png)

Focal Loss는 Cross Entropy에 $(1-p_{i})^{\gamma}$수식만 붙힌 형태이다.

단순한 저 수식이 하는 역할은 높은 예측확률에 대한 Loss값을 작게 만들어주고 작은 예측확률에 대한 Loss값은 비교적 크게 만들어 Easy Example에 대한 정확도 보다 Hard Example에 대한 정확도에 더 높은 가중치를 두어 학습을 진행하는 방법이다.

위 이미지를 통해 확인해 보면 전체적으로 Loss값이 떨어지는 것으로 보이지만 확률이 높을수록(easy example일 수록) 거의 0에 가까운 형태임을 확인할 수 있다.

![Untitled](RetinaNet%2066449fc824334333ac7ffec737b39d64/Untitled%204.png)

## Feature Pyramid Network

서로 다른 크기를 가지는 Object들을 효과적으로 Detect하기 위해 bottom up과 top down방식으로 추출된 feature map들을 lateral connection으로 연결하는 방식

![Untitled](RetinaNet%2066449fc824334333ac7ffec737b39d64/Untitled%205.png)

FPN은 이전에 YOLO v3에 대해 설명하는 글에서 언급한 바가 있으나 최초로 적용된 것은 RetinaNet이다.

FPN은 Classification모델에서 나오는 각 층의 Feature map을 이용하는 방식으로
마지막에 나온 Feature map에 1x1 conv를 적용하고 anchor box를 이용한 detection을 수행한다.

이 feature map을 upsampling하여 bottom up단계에서 이전에 나온 feature map(1x1 conv 적용)과 더하고 이를 3x3 conv를 적용하여 detection을 수행한다.

이때 결합된 feature map에 대해서 3x3 conv를 추가적으로 해주는 이유는 다른 feature map과 결합될시 잘 feature들을 잘 섞기 위함(?)이다.

위와 같은 방법을 통해 총 4개의 feature map을 추출하여 object detection을 진행하며
각각의 feature map의 pixel에대해 anchor box가 9개씩(서로 다른 크기와 스케일)생성되고 총 100k개 정도의 anchor box가 생성된다.

개별 anchor box는 Classification을 위한 k(class의 갯수)개의 확률과 bbox regression을 위한 4개의 좌표값을 가진다.
