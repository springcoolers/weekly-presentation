# 나만의 AI 비서

---

## 챗봇 시장

![1.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/12a03d61-2475-49c3-8c80-eb0231a3c5b3/1.png)

- 고객의 편리성과 회사의 경제성으로 인해 매년 전세계 챗봇 시장규모가 증가하는 추세
- precedenceresearch.com에 따르면 2023년 10억 달러(한화 약 1조 3천억원)에서 2032년 49억 달러(한화 약 6조 6천억원)로 증가할 것으로 전망
- 이는 향후 10여년동안 5배이상, 연평균 19.29%씩 증가하는 시장으로 매우 전망이 밝고 발전 가능성도 무궁무진한 분야

---

## 현재 Chat-bot의 문제점

![2.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b8a7a75a-4bcc-47b2-b207-ad85867feaa8/2.png)

- 룰베이스에 의한 기계적인 답변
- 회사들마다 통합되지 않는 다른 모델들을 제공
- Chat-gpt와 같이 성능좋은 거대 언어모델도 답변만 할뿐 직접적으로 업무를 위임하여 수행하지 않음
- 사람 마다 개인화된 모델이 아님

---

## 내가 만들 Chat-bot 비서

- 보다 자연스러운 답변
- 하나의 문제(회사)에만 적용되는 모델이 아닌 범용적으로 사용 가능한 모델
- 어시스턴스로써의 조언 뿐만 아니라 일을 위임해서 대신 하는 비서 역할을 수행
- 대화를 통해 사용자의 성향을 파악하여 지속적으로 학습(강화학습)

---

## 타임 테이블

- 문서 Q&A 및 요약 모델 학습
- Transformer 모델의 성능 향상
- 사용자의 명령에 일을 수행하는 모델 학습

---

## Transformer

### Transformer 모델

![1.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f26edbb5-1d40-4917-b408-8343a24a85da/1.png)

- Transformer 모델은 Attention만으로 이루어진 encoder-decoder 구조의 seqence to seqence 모델
- Transformer는 현재 최고의 성능(SOTA)을 자랑하는 모델 구조로서 자연어처리(NLP)뿐만 아니라 컴퓨터 비전(CV)에서도 최고 성능을 달성
- Transformer는 자연어처리 분야에서 GPT나 BERT와 같은 언어모델을 구성하는 기본적인 모델 구조 (Transformer를 여러층 쌓아 언어모델을 만듦.)

### Transformer 성능 개선 필요성

![3.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/349836b8-9327-43cc-a693-11bdda103341/3.png)

- 현재 언어모델의 파라미터수를 증가하면 성능이 좋아진다는 이론은 자연어 처리 분야에서 정설로 받아들여지고 있음
- 때문에 허깅페이스에서 공개한 그래프와 같이 최신 모델일수록 모델의 성능을 높이기 위해 모델의 크기가 기하급수적으로 증가
- 그러나 이렇게 모델의 크기를 늘려 성능을 향상시키는 방식은 기술적, 비용적 한계가 있음 (무작정 모델의 크기만을 늘릴수는 없기 때문에)
- 때문에 최근 언어모델의 가장 기본요소인 Transformer의 성능을 개선해야 한다는 목소리가 높아짐

## Next Generation Transformer

### Layer Normalization

![4.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bf0ab6f8-6b41-49f9-a6a3-6939fd52cfdd/4.png)

- Layer Normalization란 모델 입력 인스턴스(x)별로 평균을 빼주고 표준편차를 나누어주어 정규화를 수행하는 방법
- Layer Normalization을 하게되면 모델의 입력데이터값이 정규화되어 안정해짐. 때문에 오른쪽 그래프와 같이 모델의 학습이 더욱 쉽게됨.
- Layer Normalization 공식에서 x는 입력 인스턴스, E(x)는 인스턴스의 평균, V(x)는 인스턴스의 분산, gamma와 beta는 학습 웨이트를, epsilon은 분모를 ‘0’으로 만들지 않기위한 작은 상수를 나타냄.

![5.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8f163f68-14ed-4004-bfa6-e1b8db267c95/5.png)

- Transformer의 input 데이터의 분포가 불안정한 것을 파악하여, 이 부분을 lay normalization 함
- Input 지점에서 input embedding과 positional encoding 이렇게 두종류의 데이터가 만나므로 각각을 normalization 해주고 최종적으로 두 데이터를 합쳐 다시한번 normalization을 진행

![10.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/cc2afc98-0586-4ab3-b8e6-acb4e4f6b833/10.png)

- 다음 결과와 같이 layer normalization을 input 부분에 추가했을때 평균적으로 bleu 점수가 227.60%가 좋아짐

### Residual Connection

![6.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8bc2e77f-bcb3-48d0-912d-03996a1d73b8/6.png)

- 최근 거대 모델들의 등장으로 모델 layer의 깊이가 점점 깊어짐. 그러나 layer의 수가 무작정 증가하면 모델의 gradient가 ‘0’으로 수렴하거나 ‘무한대’로 발산할 수 있어 오히려 성능이 저하됨. (Gradient Vanishing과 Exploding문제 때문에.)
- 이러한 문제점을 해결하고자 ‘Residual Connection’방법론이 등장.
- ‘Residual Connection’은 input값 ‘x’를 output값 ‘F(x)’에 더해주어 모델의 gradien값을 일정수준 유지해주어 모델의 성능을 향상

![9.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d5f10ce1-a218-4a01-a468-a709b7ac26e3/9.png)

- 기존 residual connection 방식에서 “가중치를 두어 합하면 어떨까?”라는 궁금증에서 시작
- 가중치를 1에서 점점 높이자 Transformer의 성능 역시 점점 높아지다가 최고점을 찍고 다시 성능이 저하됨
- 다음 결과와 같이 평균적으로 bleu 점수가 기본 Transformer 모델에 비해 최대 159.76%가 좋아졌음

### Positional Encoding

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ffe7db8c-eecf-4090-93df-10108a26562c/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/41e020ad-7769-44d1-88a2-2bd704f2b44b/Untitled.png)

- Transformer가 RNN 시계열의 모델보다 성능이 좋은 이유는 데이터를 병렬처리하므로써 GPU사용효율을 극대화 시켰기 때문
- 자연어 데이터는 본래 시계열적 특성을 지니고 있으나 이를 병렬처리 할 시 시계열적 특성이 무시어되어 데이터의 순서를 알 수 없게 됨
- 때문에 병렬처리시에도 데이터의 순서를 알 수 있게 ‘Postional Encoding’이라는 기법이 발명
- ‘Postional Encoding’은 sin, cos의 삼각함수를 이용한 해당 단어의 위치정보를 나타냄

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ed546dba-202e-485c-bc62-b07f76e3141d/Untitled.png)

- Positional Encoding을 시각화 하면 위 그래프와 같은 모습이 됨. x축은 임베딩 벡터의 크기를, y축은 단어의 위치를 나타냄
- 보다시피 단어의 위치가 다르더라도 임베딩 벡터의 상당히 많은 부분이 동일하거나 유사하여 비 효율적

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/82ce1748-9754-4b5a-b6f4-7d6ea4b90d32/Untitled.png)

- 기존 삼각함수를 이용한 positional encoding 의 비효율적인 문제를 해결하기 위해 강화학습을 이용해 새로운 positional encoding 설계
- 강화학습 알고리즘은 최근 가장 성능이 좋다고 알려진 ＇SAC＇를 사용했으며, 거리가 가까운 단어들의 positional encoding은 서로 유사도가 높게, 거리가 먼 단어들의 positional encoding은 서로 유사도가 낮게 하기위한 강화학습 리워드를 설정
- 결과적으로 위와 같은 분포의 새로운 positional encoding 도출

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/01bbdff5-a0ab-4fae-8423-6287a69e6cf8/Untitled.png)

- 결과 새로운 positional encoding을 사용한 Transformer모델이 기존의 positional encoding을 사용한 모델보다 153.74% 성능이 향상

---

## 향후 계획

- 원래는 내가 개발한 Transformer를 이용해 GPT를 만들어 학습할 계획 이었으나, 컴퓨팅 자원의 한계로 방향 수정
- 컴퓨팅 자원이 많이 소비되는 프리 트레인이 완료된 모델을 이용
    - SKT KoGPT2
    - KakaoBrain KoGPT
- 위의 모델을 기계 독해 부분에 맞게 파인튜닝
    - Q&A 데이터셋 사용
    - KorQuAD
- 학습이 완료된 모델을 통해 원본 메뉴얼 문서에서 이용자가 찾는 답을 알려주고 이를 단계로 나눠 실행

---

## 레퍼런스 / 학습 자료

- [https://ratsgo.github.io/nlpbook/docs/language_model/tr_technics/#norm--레이어-정규화](https://ratsgo.github.io/nlpbook/docs/language_model/tr_technics/#norm--%EB%A0%88%EC%9D%B4%EC%96%B4-%EC%A0%95%EA%B7%9C%ED%99%94)
- https://korquad.github.io/
- https://github.com/SKT-AI/KoGPT2
- https://github.com/kakaobrain/kogpt
- https://arxiv.org/abs/1706.03762
- https://m.blog.naver.com/PostView.nhn?isHttpsRedirect=true&blogId=baek2sm&logNo=222176799509&categoryNo=99&proxyReferer=
