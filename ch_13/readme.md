# Segmentation

Date: 2022년 6월 6일 → 2022년 6월 10일
chapter: 13

# Segmentation

## 1.Segmentation 개요

![Untitled](Segmentation%20349a24cd56624c0fa5259cfaecffce87/Untitled.png)

Segmentation이란 이미지상에 검출된 Object에 대해서 Bounding box를 생성하는 것이 아니라 픽셀단위로 Classification하는 개념이다. Object Detection은 Object의 대략적인 위치를 나타내었다면 Segmentation은 Object의 보다 정확하고 정교한 형태를 유추할 수 있다. 

이러한 Segmentation은 2 가지로 분류할 수 있다.
(1) Sementic Segmentation과 (2) Instance Segmentation이다. 

### (1) Sementic Segmentation

![Untitled](Segmentation%20349a24cd56624c0fa5259cfaecffce87/Untitled%201.png)

Sementic Segmentation은 위 이미지에서 보이는 것과 같이 클래스에 해당되는 영역(pixel)을 분할하여 나타내는 것이다. 픽셀단위로 Classification하는 것이기 때문에 Output값으로는 각각의 픽셀에 class id가 들어있다. 

Sementic Segmentation의 단점으로는 이미지에 같은 Class에 해당하는 Object가 서로 겹쳐있다면 하나하나의 형태를 유추하는 것이 여렵다는 점이다. 

![Untitled](Segmentation%20349a24cd56624c0fa5259cfaecffce87/Untitled%202.png)

Sementic Segmentation model은 보통 위 이미지와 같은 형태를 이용한다. 
Encoder-Decoder형태이며 이미지에 Conv연산을하여 feature map의 크기를 줄이고 다시 Deconv연산을하여 원래의 형태로 복구하는 구조이다. 물론 추가적인 알고리즘이나 구조변경이 있지만 기본적인 뼈대는 위와 같다. 

### (2) Instance Segmentation

[https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FEznOQ%2FbtqCKzxqTKk%2FOxAw4ia27pjwxQ5BLryk6k%2Fimg.png](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FEznOQ%2FbtqCKzxqTKk%2FOxAw4ia27pjwxQ5BLryk6k%2Fimg.png)

Instance Segmentation은 Segmentation에서의 문제점이던 같은 Class이지만 서로다른 Instance인 경우에 각각의 형태를 알 수 없다는 점을 해결한 Task이다. 

위 이미지처럼 같은 Car Class에 해당하지만 서로 다른 Instance이기 때문에 각각의 형태를 알아볼 수 있다. 

Instance Segmentation은 Segmentation과 다르게 Two Stage의 구조로 이루어져 있으며 
Object Detection과 Segmentation이 결합되어 있다고 볼 수 있다. 

먼저 이미지에 대해서 ROI를 찾는다. 검출된 ROI에 대해서 객체가 존재하는 영역(pixel)에 Class를 주는것이 Instance Segmentation이다. 

Instance Segmentation Model인 Mask RCNN은 Faster RCNN과 FCN( Fully Convolutional Network)의 조합으로 이루어져 있으며 FCN은 Sementic Segmentation의 Model이다. 

## 2.FCN( Fully Convolutional Network for Sementic Segmentation)

> [https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf)
> 

FCN은 Sementic Segmentation의 대표적이면서 가장 기본적인 모델이다. 

![Untitled](Segmentation%20349a24cd56624c0fa5259cfaecffce87/Untitled%203.png)

FCN은 이름에서부터 알 수 있듯이 기존 Classification의 구조의 Fully Connected Layer를 Convolution연산으로 대체한 구조이다. 이러한 구조로 Feature map의 위치정보를 완전히 보전하지는 못하더라도 유지할 수 있다.

### FCN 구조

FCN의 구조는 크게 4 가지 단계로 구성된다. 

1. Feature Extractor 
2. 낮은 해상도의 Heat Map 추출 
3. Transposed Conv를 통한 Upsampling 
4. Masking 

**< 1. Feature Extractor >**

Feature Extractor부분은 다른 Classification모델과 같다. input image에 Conv연산을 하여 Feature Map을 얻는 것을 목적으로 한다. 

**< 2. Heat Map 추출 >** 

Feature Extractor를 통해 나온 Feature Map을 토대로 7x7x(class) 크기로 1x1 Conv연산을하여 Heat Map을 추출한다. 

기존의 classification모델은 이 부분이 fc 로 되어있던 반면 FCN에서는 1x1 conv연산을 하여 위치정보를 가져간다는 것이 FCN의 핵심적인 아이디어이다. 

이때, 최종적인 Feature map의 channel의 크기는 Dataset의 Class 수로 설정되며 각각의 Channel은 각각의 Class일 확률을 담은 Pixel들로 이루어 진다. 예를 들어 위 이미지에서는 최종 Feature Map의 고양이에 해당하는 Channel을 시각화 해보면 input image에서 고양이가 있는 영역만 활성화된 것을 확인할 수 있고 각각의 Pixel에 확률값이 들어가 있다는 것을 알 수 있다. 

**< 3. Transposed Conv >**

위 두 가지 과정을 통해 얻어낸 Feature map을 Upsampling하여 원본의 크기로 복구하여야 한다. 하지만  단순히 x32를 하면 정사각형의 형태로 Segmentation될테니 Transposed Conv를 한다. 

![https://user-images.githubusercontent.com/50395556/81541105-9401e980-93ad-11ea-87a1-a7676fbd8314.png](https://user-images.githubusercontent.com/50395556/81541105-9401e980-93ad-11ea-87a1-a7676fbd8314.png)

Transposed Conv란 기존의 Conv연산의 반대개념으로 Feature map에 Filter를 연산하여 더 큰 형태의 Feature map을 만드는 것이다.(겹치는 부분은 더함) 이 과정에서 사용하는 Filter는 학습되는 가중치이다.

<aside>
💡 Transposed Conv vs DeConv
- Transpsed는 위에서 설명한대로 Feature map에 새로운 filter를 연산하여 더 큰 feature map을 만드는 반면 DeConv는 이전에 사용한 Filter를 사용하여 원래의 형태를 복구하는 구조이다. 
따라서 두 Conv사이에는 Transposed는 filter를 새로사용하며 학습되는 가중치라는 차이점이 있다.

</aside>

**< 4. Masking >**

위 결과로 만들어진 feature map에 각각의 해당하는 class별로 Masking을 하여 return 하면 분할이 완료된다. 

### **Skip Combining**

![Untitled](Segmentation%20349a24cd56624c0fa5259cfaecffce87/Untitled%204.png)

위에서 알아본 Transposed Conv를 이용한 Upsampling을 단순히 진행하게 된다면 위치정보의 부족으로 인해 원하는 결과를 내기가 힘들었다. 따라서 최종적인 Feature Map에 손실된 위치정보를 더하기 위하여 Skip Combining을 적용한다. 

논문에 기재된 바로는 32s/16s/8s 의 3 가지로 나타나 있으며 8s가 가장 성능이 뛰어나다. 

32s의 경우 최종 Feature Map을 Upsampling한 그대로를 의미하고 
16s의 경우 최종 Feature Map을 2배 Upsampling한 후에 이전 pooling 단계의 Feature map과 더한 후에 Upsampling 한 것이다. 이러한 과정을 통해서 손실된 위치정보를 어느정도 복구할 수 있었고 
8s의 경우 더한 Feature Map을 2배해 또 이전 단계의 Feature map과 더하여 Upsampling 한 것이다. 

![Untitled](Segmentation%20349a24cd56624c0fa5259cfaecffce87/Untitled%205.png)

결과를 참고해 보면 8s가 가장 성능이 좋으며 Skip Combining이 위치정보를 더하는데 효과가 있었음을 알 수 있다. 

FCN의 문제점은 지나치게 Upsampling 단계에 의존적이었다는 것이며 이후에 나온 논문들은 Upsampling과정을 보완하는 것에 집중하여 성능을 개선하였다.