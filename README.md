# final-project-level2-recsys-13

## Project Overview
### DocVQA
- 문서 이미지로부터 기계가 읽을 수 있는 형태로 정보를 추출하여 주어진 질문에 대한 답을 하는 작업

### Metric
- ANLS(Average Normalized Levenshtein Similarity)

### Dataset
- UCSF Industry Documents Library의 문서를 사용
- 12,767개의 이미지에 대한 50,000개의 질문으로 구성
- 80/10/10의 비율의 train, valid, test로 분할
- 9개의 질문 카테고리가 존재
    - handwritten, form, table/list, layout, running text, photograph, figure, yes/no, other
    
### Model
- LayoutLMv2
![Untitled (14)](https://user-images.githubusercontent.com/64139953/217515568-c02e0ca7-073c-4343-86b5-23cfaed7b7e6.png)

## Result
