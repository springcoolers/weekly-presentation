## 개요

1. 나무위키 데이터를 사용해서 지식그래프 만든다. (정형화된 infobox만 사용. 설명글은 안 쓸 것임.)
    1. 데이터 출처: https://mu-star.net/wikidb (2021.3.1 까지의 데이터 받을 수 있음)
    2. 2주동안 해보고 잘 안되면 이미 있는 영어 지식그래프 쓰기(dbpedia 또는 freebase) 파싱 너무 복잡하면 그냥 패스… 
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/67e4ec0e-b073-4ee9-9aae-f09f5999d37a/Untitled.png)
    
    - 이렇게 생긴 부분만 사용할 예정.
2. 지식그래프를 neo4j에 적재한다.
3. chatGPT api를 사용하여 영어 질문을 넣어서 neo4j의 쿼리인 cypher로 바꿔서 db에 쿼리 날려서 정답 가져온다. 
    1. 질문 예시: sbs에서 방영한 김은숙 작가의 드라마에 출연한 배우가 누구야

## 주제 선정 이유

- chatGPT의 한계: 정보의 오류
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b02a7d5f-047b-45d7-adae-26968f63ef18/Untitled.png)
    
    - 가을동화, 사랑의 불시착은 김은숙 작가의 작품이 아님. 사랑의 불시착은 sbs가 아님
- 지식그래프는 ground truth 정보만 가지고 있기 때문에 쿼리만 잘 던지면 정답은 올바르게 나온다. 다만 자연어를 넣었을 때 db query로 바꾸는 작업이 매우 어렵다.
- chatGPT가 바로 질문에 답하게 하면 아직은 오류가 많지만, 지식그래프에 던질 수 있는 쿼리를 잘 만들게 하면 그걸 활용해서 정확도 높은 정답을 가져올 수 있다.
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/01ec5058-89c7-4d93-9766-1f896c55d3c0/Untitled.png)
    
- 공개된 한국어 지식그래프가 없는데 나무위키로 만들어보면 재밌을 것 같다.
