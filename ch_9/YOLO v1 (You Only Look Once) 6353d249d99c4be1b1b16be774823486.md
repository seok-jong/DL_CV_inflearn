# YOLO v1 (You Only Look Once)

Date: 2022년 5월 16일 → 2022년 5월 20일

# YOLO

![https://pjreddie.com/static/img/darknet.png](https://pjreddie.com/static/img/darknet.png)

Yolo는 Darknet이라는 C로 작성된 딥러닝 프레임워크를 기반으로 만들어진 Object Detection모델이다. Darknet은 Yolo의 주요 저자인 Joseph Redmon이 독자적으로 만든 프레임워크이다. 

이러한 Darknet이 최초로 적용된 모델이 Yolo v1이며 이 모델은 최초의 최초의 One-stage Detector 모델이 된다.

Yolo의 버전별 특징은 다음과 같다. 

- v1 : 빠른 Detection 시간 그러나 낮은 정확도
- v2 : 수행 시간과 성능 모두 개선 , SSD에 비해 작은 Object 성능 저하
- v3 : v2 대비 수행 시간은 조금 느림 , 성능 대폭 개선
- v4 : v3 대비 수행 시간 약간 향상 , 성능 대폭 개선

이렇듯 Yolo는 꾸준히 발전하는 형태로 업그레이드 되어 왔다. 

## YOLO v1

![Screenshot from 2022-05-20 18-28-54.png](YOLO%20v1%20(You%20Only%20Look%20Once)%206353d249d99c4be1b1b16be774823486/Screenshot_from_2022-05-20_18-28-54.png)

Yolo v1은 입력 이미지를 **S x S Grid**로 나누고 각 **Grid의 Cell이 하나의 Object에 대한 Detection을 수행하는 구조**이다. 
각 Grid Cell이 2개의 Bounding Box후보를 기반으로 Object 의 Bounding Box를 예측하는 구조이다. 

 이러한 구조는 Object Detection이 한 번에 진행된다는 이점을 가져왔고 그에따라 속도에 대한 향상까지 챙길 수 있었다. 

하지만 Cell을 중심으로 Object Detection을 진행하기 때문에 위 이미지처럼 하나의 cell에 Object가 여러개가 포함되어 있는 경우에도 1개의 Object만 Detect할 수 있었다. 

### YOLO v1  네트워크 및 Prediction 값

![https://www.harrysprojects.com/images/articles/yolov1/architecture.png](https://www.harrysprojects.com/images/articles/yolov1/architecture.png)

Yolo v1의 네트워크 구조는 위 이미지와 같다. 

448x448x3크기의 input 이미지를 feature extractor모델(Inception모델을 변형한 형태 사용)과 Fully Connect를 통해 **7x7x30**의 크기의 Feature Map을 만들어 낸다. 

여기서 7x7x30은 최종적으로 classification과 regression을 하기위한 feature map이라고 볼 수 있다. 

7x7은 Grid Cell의 크기를 의미하고 channel로는 각각의 Cell에 해당하는 정보를 담고 있다. 

30은 ( B x 5 + C)를 의미하는데 

B : Bounding Box의 갯수( 각 Grid Cell별) 

C : Class의 갯수 ( Pascal VOC의 경우 20개 ) 

channel의 데이터를 좀 더 자세히 보자면 

1개의 Bound Box에 대해서 5개의 데이터가 존재하는데( B x 5 ) [ x, y, w, h, Confidence ]를 의미한다.
여기서 Confidence는 어떠한 객체일 확률이 아니라 단지 **“어떠한 Object일 확률” x IoU값** 이다. 

이러한 정보가 2개 존재하고 나머지 20은 Class별 확률값이다. 
이때 2개의 Bounding Box중 Confidence가 더 높은 Bbox에 대한 확률이다. 

### Loss Function

[https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbatH5J%2FbtqVbwdxWo9%2Fyon75K37PSl84r9zmK76a1%2Fimg.png](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbatH5J%2FbtqVbwdxWo9%2Fyon75K37PSl84r9zmK76a1%2Fimg.png)

Yolo v1의 Loss Function은 위 이미지와 같이 구성되어 있다. 

크게 4개의 부분이 더해져서 최종 Loss Function을 이루게 되는데 부분 부분 살펴보자. 

**< BBox 중심 x, y 좌표 Loss >**

![Untitled](YOLO%20v1%20(You%20Only%20Look%20Once)%206353d249d99c4be1b1b16be774823486/Untitled.png)

$\lambda$ 는 x, y 좌표를 학습시키기위해 발생되는 손실값에 얼만큼의 영향을 줄건지에 대한 가중치이다. 

$\Sigma$ 는 좌측부터 모든 셀에 대해서 고려하는 것이고 우측은 모든 Bbox에 대해서 고려한다는 의미이다. 

$x$는 예측한 Bbox의 중심 x좌표를 의미하고 $\hat{x}$ 는 GT의 중심 x좌표를 의미하며 이는 y에도 똑같이 적용된다. 

$1_{ij}^{obj}$는 Object를 책임지는 BBox만 고려한다(아니면 0으로 계산)는 뜻인데  cell을 7x7로 나누고 cell당 bbox를 2개씩 고려한다면 bbox는 총 98개 생성되는데 이 중 의미있는 값에 대해서만 고려한다는 의미이다. 

**< BBox 너비 w, 높이 h Loss >** 

![Untitled](YOLO%20v1%20(You%20Only%20Look%20Once)%206353d249d99c4be1b1b16be774823486/Untitled%201.png)

너비와 높이의 경우에도 x,y 좌표에 대한 loss값을 구할때와 유사하다.
다만, 제곱근 계산을 하는 이유는 큰 Object와 작은 Object와의 편차를 줄이기 위해서이다. 
크기가 큰 Object일 경우 Loss값이 비교적 커질 수 있기 때문이다. 

**< Object Confidence Loss >**

![Untitled](YOLO%20v1%20(You%20Only%20Look%20Once)%206353d249d99c4be1b1b16be774823486/Untitled%202.png)

$C_{i}$는 **Model이 예측한 Object일 확률 * IoU값**이고, $\hat{C_{i}}$는 GT값이며 1이다.

4번에 해당하는 Loss값은 객체에 대해서 대표가 될 수 없으나 object를 검출한 box에 대해서 패널티를 주는 값이다. 

**< Classification Loss >** 

![Untitled](YOLO%20v1%20(You%20Only%20Look%20Once)%206353d249d99c4be1b1b16be774823486/Untitled%203.png)

classification에 대한 Loss값은 좌표에 대한값은 무시하고 각각의 Cell에 대해서 class가 맞는 경우만 고려하여 Loss값을 계산한다. 

### NMS

![Untitled](YOLO%20v1%20(You%20Only%20Look%20Once)%206353d249d99c4be1b1b16be774823486/Untitled%204.png)

Yolo와 같은 One-stage Detector는 Bbox를 많이 찾는다는 특징이 있다. 
위 이미지처럼 많은 수의 bbox를 찾으면 이를 NMS를 통해 거르는 작업을 한다. 

< 개별 Class별 NMS수행 > 

1. 특정 confidence 값 이하는 전부 제거 
2. 가장 높은 Confidence를 가진 순으로 BBox정렬 
3. 가장 높은 Confidence를 가진 BBox와 겹치는 부분이 많은( IoU Threshold를 넘는) BBox를 전부 제거 
4. 3번 반복 

### Limitation

- 수행시간은 줄어들었지만 성능이 안좋음
- 특히 작은 Object에 대한 성능이 더욱 안좋음
- Cell 기반 Detect를 수행하기 때문에 한 Cell에 여러개의 Object가 있는 경우 모든 Object를 Detect하는 것이 어렵다.

<aside>
🔥 1. 7x7x30에서 confidence 가 작은 값의 5개의 데이터는 왜 가지고 있으며 이 값들이 Backprop당시에 영향을 주거나 받는지

</aside>

---

### Reference

[https://visionhong.tistory.com/15](https://visionhong.tistory.com/15)

[https://www.harrysprojects.com/articles/yolov1.html](https://www.harrysprojects.com/articles/yolov1.html)