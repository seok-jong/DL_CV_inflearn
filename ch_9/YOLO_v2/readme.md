# YOLO v2

# YOLO v2

![Untitled](YOLO%20v2%209a094d55ed83463888a8cc7389533e21/Untitled.png)

YOLO v2는 SSD가 만들어진지 1년 후인 2016년 12월에 발표되었다. 
v1에 비해 달라진 점을 보자면 Backbone모델을 Darknet 19를 사용하였고, Anchor Box를 사용하였다. 
가장 큰 변화라고 할 수 있는 건 Anchor Box의 결정 방법에 K-Means Clustering을 사용하였다는 것이다.

## 특징

- Batch Nomalization 적용
- High Resolution Classifier : 448 x 448 크기의 이미지를 이용하여 Classifier 부분을 fine tuning
- 13 x 13 Featur Map 기반에서 개별 Grid Cell별 5개의 Anchor Box에서 Object Detection
- Direct Location Prediction 적용
- Darknet 19 모델 적용
- Classification layer 부분에서 Fully Connect부분을 Fully Convolution으로 변경하고 다양한 크기의 이미지로 학습

## Anchor Box

![Untitled](YOLO%20v2%209a094d55ed83463888a8cc7389533e21/Untitled%201.png)

v2는 SSD와 마찬가지고 각각의 Cell에서 여러개의 Anchor박스를 생성하여 이를 기준으로 Object Detection을 수행한한다. 

또한 K-Means Clustering을 통해 데이터세트의 크기와 ratio를 5개의 군집화 분류를 통하여 Anchor box를 계산하였다. 

## Output Feature Map

![Untitled](YOLO%20v2%209a094d55ed83463888a8cc7389533e21/Untitled%202.png)

v2에서의 Output Feature Map의 형태이다. 

13x13으로 크기(Cell의 갯수)를 늘렸고 Depth는 각각의 Anchor Box에 대한 정보들을 담고있다. 

이전의 v1에서는 bbox 2개에 대한 Confidence와 좌표, 그리고 confidence가 높은 bbox에 대한 class별 score가 담겨있었던 반면 
v2에서는 모든 Anchor box에 대한 좌표와 confidence, class별 score가 담겨있다. 
따라서 Pacal VOC기준으로 {4(좌표) + 1(Confidence) + 20(Class Score)} * 5(Anchor Box) 의 깊이를 가지는 Feature Map이 만들어 진다. 

## Direct Location Prediction

![Untitled](YOLO%20v2%209a094d55ed83463888a8cc7389533e21/Untitled%203.png)

Direct Location Prediction은 예측한 값이 Cell에서 너무 벗어나지 않도록 하는 방법이다. 

- t_x, t-y : 모델이 실제로 예측한 x,y좌표의 offset
- t_w, t_h : 모델이 예측한 높이, 너비
- c_x, c_y : Cell의 값이며 좌상단부터 교차점을 중심으로의 좌표이다.
- $\sigma$ : 시그모이드함수이며 0~1값으로 scale을 조정해 준다.
- p_w, p_h : Anchor box의 높이, 너비

b_x,b_y를 구할 때, $\sigma$계산을 하는 것이 이 알고리즘의 핵심이라 할 수 있는데, 예측한 offset이 아무리 크게나와도 1을 넘지 않게 함으로써 Cell 밖으로 좌표가 나가지 않게 한다. 

## Loss Function

![Untitled](YOLO%20v2%209a094d55ed83463888a8cc7389533e21/Untitled%204.png)

v2에서는 v1과 같은 Loss Function을 사용한다. 
 다만 예측값과 Confidence 값을 넣어줄 때 고려해야 할 점이 있다.

예측 좌표값과 높이 , 너비값은 이전의 Direct Location Prediction을 통해 계산한 값을 넣어줘야 한다. 

Confidence Score의 경우, x,y좌표를 계산할때와 마찬가지로 sigmoid 계산을 적용하여 사용하게 된다.  

## PassThrough module

![Screenshot from 2022-05-20 23-49-16.png](YOLO%20v2%209a094d55ed83463888a8cc7389533e21/Screenshot_from_2022-05-20_23-49-16.png)

작은 크기의 오브젝트를 Detect 하기 위해 26 x 26 x 512 크기의 Feature Map의 특징을 유지한 채, 
13 x 13 x 2048로 reshape하여 13 x 13 x 1024에 추가한다. 

이러한 작업이 작은 Object를 Detect하는데 도움을 주는 이유는 
Feature Map의 형태가 갈수록 작고 길어지면서 세세한 Feature보단 전체적인 패턴을 보유하게 된다. 
그래서 덜 작아진 형태의 Feature Map(아직까지는 세세한 특징을 보유하고 있는)을 뒤로 한번 더 상기시켜주는 느낌으로 붙여주게된다. 

## Multi-Scale Training

![Screenshot from 2022-05-20 23-55-02.png](YOLO%20v2%209a094d55ed83463888a8cc7389533e21/Screenshot_from_2022-05-20_23-55-02.png)

v2로 오면서 달라진 점에서 언급했던게 Classification layer를 Fully connected가 아닌 Fully Convolution layer로 사용하였다는 것이었다. 

따라서 동적으로 입력 이미지의 크기변경이 가능했고 이러한 점을 이용하여 Training시에 10회 배치마다 입력 이미지 크기를 모델에서 320부터 608까지(32의 배수) 동적으로 변경하여 진행하였다.

이러한 학습방법은 inference시에 다양한 이미지에 대해서 모델이 robust하게 동작하도록 만들었다.