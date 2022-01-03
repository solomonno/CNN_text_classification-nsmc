# Description
- CNN(Convolutional Neural Network)을 이용한 텍스트 분류
- numpy, pandas를 이용하여 data preprocessing
- pytorch를 이용하여 모델 구현

## Dataset
### KLUE ynat dataset
- 뉴스 기사 제목 분류 데이터셋(train: 45678, dev: 9107)
- `title`: 뉴스 기사 제목, `label`: 기사 분류 카테고리
- train을 trainset으로, dev를 testset으로 모델학습

### NSMC dataset 
(TO DO)

## Model
### CNN-rand
- embedding layer의 weight를 random하게 initialization

### CNN-static
- pretrained embedding layer를 사용
- embedding layer의 weight를 학습하지 않음

### CNN-non-static
- pretrained embedding layer를 사용
- embedding layer의 weight도 학습

---
# Dependencies
(TO DO)

---
# Result
dataset|CNN-rand|CNN-static|CNN-non-static|
---|---|---|---|
ynat|-|-|-|
NSMC|-|-|-|


---
# References
- [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
- [KLUE ynat dataset](https://github.com/KLUE-benchmark/KLUE)
- [NSMC dataset](https://github.com/e9t/nsmc)
