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

y자리에 실제값이 들어가며 실제값이 1 일 경우에만 loss 값에 영향을 주는 방식이다. 

-ln그래프를 참고해 보면 x(p)의 값이 작을수록 -$\infty$에 가까운 값이며 이는 실제 정답인데 예측확률(softmax의 결과)이 작을 때에 해당된다. 

반대로 p의 값이 크면 0에 가까워지며 loss값에 덜 영향을 주게 된다. 

따라서 Cross Entropy방식은 모델이 정답을 더욱 정확하게 잘 맞추는데 초점을 맞추는 Loss Function이며 정답이 아닌경우에 확실히 아니라고 하는데에는 초점을 맞추고 있지 않다. 

### **< One-Stage Object Detector의 Class Imbalance문제 >**

![Untitled](RetinaNet%2066449fc824334333ac7ffec737b39d64/Untitled%202.png)

train image의 대부분이 Object보단 background 영역이 대부분이다.  
이러한 Background에 대한 Loss 값은 크지 않지만 그 수가 굉장히 많기 때문에(One-Stage Detector의 특징인 Anchor box를 사용하기 때문) 객체에 대한 Loss값보다 더욱 도드라지게 된다. 

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