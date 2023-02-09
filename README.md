# final-project-level2-recsys-13

## DocVQA

### DocVQA
- 문서 이미지로부터 기계가 읽을 수 있는 형태로 정보를 추출하여 주어진 질문에 대한 답을 하는 작업
![image](https://user-images.githubusercontent.com/64139953/217688759-0161dfad-611b-4a4d-b7d7-384ac714308a.png)


### Metric
- **ANLS(Average Normalized Levenshtein Similarity)**
    - N: 문제의 전체 수
    - M: 각 문제 당 답변의 수
    - $a_{ij}$: i번 문제의 j번째 답변
    - $o_{q_{i}}$: i번 문제의 예측 답변
    - NL: 두 문자열 사이의 Normalized Levenshtein distance
    
![image](https://user-images.githubusercontent.com/64139953/217688837-317f5b54-a137-4a3c-a387-a925303cb570.png)


### Dataset
- UCSF Industry Documents Library의 문서를 사용
- 12,767개의 이미지에 대한 50,000개의 질문으로 구성
- 80/10/10의 비율의 train, valid, test로 분할
- 9개의 질문 카테고리가 존재
    - handwritten, form, table/list, layout, running text, photograph, figure, yes/no, other
![image](https://user-images.githubusercontent.com/64139953/217688934-2819bb50-a3f1-44eb-aa1a-acbe41fd3fbf.png)


## 모델 및 파이프라인
### LayoutLMv2
- 텍스트와 이미지 정보 뿐만 아니라 bounding box값을 의미하는 layout 정보를 embedding에 포함한 모델
![image](https://user-images.githubusercontent.com/64139953/217689055-410c9424-e24a-4382-95f2-848399015015.png)


## 카테고리 별 에러 분석
### **Table**
**Question**: How many employees are in united states in the year 1970?
<br> **Detection**: 684,383
<br> **Ground Truth**: 14,400

**문제점**
: row와 column이 교차되는 곳의 텍스트 정보를 제대로 가져오지 못함

**해결 방안**
: 질문 데이터를 직접 생성하여 추가 학습

**방법**
- row와 column이 교차하는 지점에서 값을 가져오는 문제는 3가지 이상의 sub-task로 이루어져있다고 생각했음
    1. row 찾기
    2. column 찾기
    3. row와 column 교차하는 부분 찾기  
- 따라서 row와 column을 먼저 찾을 수 있어야 함
- 데이터를 확인해보니 train 기준 row 혹은 column 단어가 포함된 데이터는 0.2%에 불과
- 다음과 같은 방식으로 column에 대한 질문 데이터만 200개 가량 생성했음
- what is the name of the {first/second/third, ...} column of the table?    

### Yes/No

![jlmd0217_2.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ae0eea8f-89af-4ada-8d49-2ce8cde9653f/jlmd0217_2.png)

**Question** : Was the event sponsored by a Pharmaceutical company? 

**Detection** : pfizer

**Ground Truth** : [Yes]

**문제점**

: 답을 yes 또는 no 중에서 값을 가져와야 하는데 다른 값을 예측

**해결 방안**

: yes or no /A or B 유형의 질문인 경우 yes, no / A, B에서 답을 가져오도록 학습

- **방법**
    
    **배경**
    
    - 질문의 답이 yes/no 이거나 A/B인 경우에는 다른 질문들과 다른 방법으로 접근
    
    **방법**
    
    - yes/no 이거나 A/B 인 질문들의 특징을 이용해서 분류 후 질문에서 답을 찾는 모델 생성
    - yes/no 이거나 A/B 문제의 특징
        - 질문이 when, where, why, what, who, how 로 시작하지 않음
        - is, are, was, were 등의 be 동사로 시작
        - 이후 질문의 특징을 활용해서 데이터를 필터링한 후에 질문에서 답을 extract 하는 모델을 생성
    
    **결과**
    
    - Yes/No 카테고리의 데이터가 충분하지 않아 성능 개선에 실패
    

## 추가 성능 개선 시도

### **OCR 결과 정렬**

**배경**

- Sequential 모델을 사용했기 때문에 텍스트의 순서가 중요함
- 단순히 위에서 아래로 읽지 않고 문서의 단락을 고려해서 각각 읽는 것이 텍스트를 이해하는데 더 도움이 될 것이라 판단했음

**방법1**

- 각 단어 사이의 gap이 특정 크기 이상인 경우 두 단어를 서로 다른 단락으로 판단하여 순서를 변경
- 고정된 gap을 사용하여 단어의 순서를 변경
    - gap = 16, 18, 20, 22, 24, 26, 28
- 각 단어의 글자 크기를 기준으로 gap을 설정, 단어의 순서를 변경
    - gap = 글자 크기 * (1 / 1.5 / 2)

**방법2**

- Layout Reader 모델 사용

### **Fuzzy Matching**

**배경**

- Extraction based 모델을 사용했기 때문에 주어진 context에서 answer에 해당하는 부분의 start, end index를 각각 찾아야했음
- 이를 위해 처음에는 단순한 Exact matching 방법을 적용했지만 train set에서 약 24%의 데이터에서 각 index를 찾지 못했음
- index를 찾지 못한 데이터를 확인해본 결과, answer: ‘b&w headquarters’, match: ‘baw headquarters’ 와 같이 OCR 모델이 &를 a로 판단하는 등의 문제를 발견했음
- 이를 보완하기 위해 전체 문자에서 한 글자, 두 글자 정도는 달라도 같은 문자로 인식하도록 하는 아이디어에서 Fuzzy Matching 방법을 적용했음

**방법**

- ratio가 0.8 이상일 때만 같은 문자로 간주

### **Question Paraphrasing**

**배경**

- Train set 기준 question을 구성하는 word의 75%가 3번 이하의 빈도를 가짐
- 생소한 단어일수록 모델이 그 의미를 해석하기 어려울 것이라 판단함

**방법**

- T5 기반의 paraphraser 모델을 사용하여 질문의 단어를 변환
- 생소하고, 다양한 단어 빈도를 줄이는 것이 목적이므로 모델의 파라미터 diversity를 0으로 설정하여 표준화하고자 함

### Finding Machting Index

**배경**

- Train, Valid set에서 주어진 문제에 대한 정답을 문서 내에서 찾는 경우 matching이 유일하지 않은 경우가 존재하여 Baseline에서는 단순히 가장 처음 match된 index를 반환
- 하지만 처음 match된 부분이 실제 정답과 값만 같은 경우들이 존재하기 때문에 정확한 학습을 위해서 아래의 2가지 방법을 사용하여 실험을 진행

**방법**

- Matching된 모든 index를 학습에 사용
- Matching된 index 중 해당 문장의 앞, 뒤의 문장과 질문의 유사도가 최대인 index만을 학습에 사용

### **Amazon Textract**

**배경**

- OCR이 제대로 수행되지 않은 문서가 다수 존재

**방법**

- Amazon Textract를 train/valid set 약 1000개와 전체 test set에 적용

### **Stride를 활용하여 긴 sequence 자르기**

**배경**

- token의 길이가 max length를 넘어가는 sequence와 같이 답을 찾지 못하는 경우 존재

**방법**

- Stride를 활용해서 여러 개의 sequence로 분할

### **Large Model**

**방법**

- Parameter가 더 많은 모델을 사용하여 성능 향상을 시도
    - LayoutLMv2BASE : 12-layers, 12-heads, hidden size = 768
    - LayoutLMv2LARGE : 24-layers, 16-heads, hidden size = 1024

## 결과

### 실험 결과

### 최종 결과

## 한계점 및 개선방향

### **한계점**

- OCR의 한계로 추가적인 문서 데이터를 활용할 수 없었음
    - BenthamQA, SROIE datasets 등의 데이터셋을 이용해 추가 학습을 진행하고자 하였으나 OCR 사용 시 비용 상의 문제로 진행하지 못함
- 데이터셋에 카테고리에 대한 라벨링 값이 없어 카테고리 분류에 어려움을 겪음
- LayoutLMv2 모델의 경우 이미지 정보를 잘 활용하지 못하여 figure, image와 같은 카테고리에 취약함
- 다른 여러 모델들을 실험해 앙상블하지 못한 점

### **개선 방향**

- 문서의 카테고리를 분류할 수 있다면 각 카테고리에 특화된 모델 적용
    - figure와 image와 같은 카테고리에는 LayoutLMv2가 아닌 다른 모델을 통한 학습 진행
- 각 카테고리별, 모델별로 나온 결과를 앙상블 진행

## Result
![image](https://user-images.githubusercontent.com/64139953/217686986-fccc80e9-9c78-4d1b-84a1-36a1db68cf9e.png)



