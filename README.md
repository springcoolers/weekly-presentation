
# (2023 봄 Pseudolab Openlab 6기) GitHub에 NLP 잔디심기 2👋

# This repo is : 스터디에서 활용되었거나 개인적으로 공유하고자하는 코드를 공유하는 곳으로 활용됩니다.

🔭 스터디 일정은 이 링크를 확인하세요
https://pseudo-lab.com/NLP-2-c5158177879c4bcab6e4106c053b44f5




<aside>
💡 이 Openlab은 뭘 하나요?

</aside>

> 실생활에 쓸 수 있는 유용한 nlp 플젝을 만들면서 모르는 것 찾아보고 공부하는 스터디
> 

> 자신이 만들고 싶은 NLP 아이디어를 구현한다. 구현한 내용을 다른 사람들에게 공유한다.
> 

> 각자 프로젝트를 정하고 이를 다른 구성원들에게 발표를 통해 소개 및 토론을 합니다.  또한 최종적으로 프로젝트를 완수하면 끝. "작아도 좋으니깐 꾸준히 하자”
> 

> NLP를 활용한 느슨한 프로젝트 구현이 핵심으로, 서로 관심 있는 주제를 공유하고 공부하며 동기부여하는 모임이라고 이해했습니다.
> 

→ 다 맞습니다! 

### 지난 시즌 내용 요약:

---

- Basic Tasks: NER, 요약, 분류, 생성 등
- 허깅페이스 프레임워크 사용의 기초
- tweaking : 모델 변경, 데이터셋 바꾸기 등 간단 응용

### 이번 시즌 목표:

---

<aside>
💡 의식적 훈련: 내 toolbox → 현실 문제 풀기

</aside>

1. 현실 문제 찾기 
2. 해결책을 찾고, 모방해서 구현해보기 (클론 코딩)
3. ultimately: 직접 하나 만들고, huggingface에 올리기 → 프리시즌 이후. 매우 장기적 목표 

ex) 반복문 → max number of island → 스크린골프 계산 프로그램 

### 1# Ideation - 사람들은 NLP로 뭘 만들까?

- sources &  설명
    1. Datacamp
    2. kaggle 
    3. Reddit - showcases
    4. Huggingface Spaces
    5. Youtube, blogs
    6. 논문, 학회지
    7. conference : [deview](https://deview.kr/2023)
    8. indiehackers / [futurepedia.io](http://futurepedia.io) 
    
    아래로 갈수록 찾기 힘들고, 전문적이고, 난이도가 높아진다
    

내가 만든 프로그램! (link) 

### 2# 만들기 (힘 닿는 곳 까지)

ex) datacamp로 예시 보이자

발표 규칙

- 하나씩 print를 찍어주면서, 데이터 형태를 보여주면 좋겠다.
- 관련된 이론들 - 상세히 발표

### 3# 자료 정리 → 방안

- 발표 - 녹화해서 가짜연 유튜브에 올릴까 함
- code - github
- **Build on Public**
- 찾은 Idea 정리 - Repo / 블로그 글
    - (한다면) 이건 별개로 하고싶다.
- 노션: 자료 아카이브
    1. 발표 자료 (내가 복사해서 넣을게요!)
    2. 모르는거 찾아보기 → 이 부분을 리서처분들, 학생분들이 도와주시면 좋겠습니다. 

- 찬란님이 올려주신거
    - 깃헙 (프로젝트, 팀, 리포)
        - 초대
            
            [가짜연구소 (Pseudo Lab)](https://github.com/Pseudo-Lab?q=book&type=all&language=&sort=)
            
        - 팀
            
            [](https://github.com/orgs/Pseudo-Lab/teams)
            
        - 프로젝트
            
            [GitHub](https://github.com/orgs/Pseudo-Lab/projects)
            
        - 리포
            
            [https://github.com/Pseudo-Lab/nerds-nerf](https://github.com/Pseudo-Lab/nerds-nerf)
            
    - 자료 공유
        - 노션
        - 깃헙
            - Jupyterbook
                
                [Jupyter Book — PseudoLab Jupyter Book Tutorial](https://pseudo-lab.github.io/Jupyter-Book-Tutorial/intro.html)
                
                [https://github.com/Pseudo-Lab/Jupyter-Book-Template](https://github.com/Pseudo-Lab/Jupyter-Book-Template)
                
                [https://github.com/Pseudo-Lab/Tutorial-Book](https://github.com/Pseudo-Lab/Tutorial-Book)
                
                [https://pseudo-lab.github.io/Tutorial-Book/](https://pseudo-lab.github.io/Tutorial-Book/)
                
                ![Untitled](%E1%84%89%E1%85%B5%E1%84%8C%E1%85%B3%E1%86%AB%203%20OT%20%E1%84%89%E1%85%B3%E1%84%90%E1%85%A5%E1%84%83%E1%85%B5%20%E1%84%89%E1%85%A9%E1%84%80%E1%85%A2%20%E1%84%87%E1%85%A1%E1%86%BC%E1%84%92%E1%85%A3%E1%86%BC%20%E1%84%82%E1%85%A9%E1%86%AB%E1%84%8B%E1%85%B4,%20Ice%20Breaking%20%20ed556527d5944775863c60f9e58b5de4/Untitled.png)
                
            - 코드
        - 블로그
            - Blog Footer (업데이트 예정)
                
                이 포스트는 2023년도 상반기에 진행된 가짜연구소 아카데미 프로그램의 일환으로 작성하였습니다. 지식, 정보, 교육의 격차를 줄이며 공유, 동기부여, 함께하는 즐거움의 가치를 전파하는데 기여하게 되어 기쁩니다. 가짜연구소 활동에 관심이 있으신 분들은 [웹사이트](https://pseudo-lab.com/)나 [디스코드 커뮤니티](https://discord.gg/EPurkHVtp2)를 살펴보세요!
                
                ---
                
        - Youtube
            - 가짜연구소 공식 채널: [https://www.youtube.com/channel/UCLxNgQ_Ir6Cuod9mkBBiPEw](https://www.youtube.com/channel/UCLxNgQ_Ir6Cuod9mkBBiPEw)
            - 개인 채널 + 재생목록, 태그 추가(#가짜연구소 #PseudoLab #MLPaperReadingClubs)

### 4# 채널

디스코드 : 빌드 온 퍼블릭 뭔가 만드는 이야기는 여기서

### 5# 참여 형태

- n회이상 참여시 멤버 -
- 멤버란 뭘까요?
    - 계속 나와주시기로 약속한 사람들
    - 수준을 맞추고
    - 서로에게 관심을 줘야 합니다
    - 적게 뽑은것도, 저 ↔ 참가원 : 다 케어할 수가 없었음

- 참여 형태: 사다리 vs 청강 - 이제는 팀 멤버여야 할 필요가 없다.
    - 그냥 매주 편하게 와 주셔도 좋습니다
    - 스펙, 트랙 레코드 남기려는 목적 아니면
    - **기획**을 내면 계획표에 넣고 발표로 참여하는걸로, 아니면 그냥 편하게 와서 놀다 가시라

팀별 조정기간 안내ㅐ

- 팀별 조정 기간
    
    팀별 조정기간 빌더의 역할은 다음과 같습니다.
    
    1. OT 진행
    2. 러너 추가 선정
    
    팀별 조정기간에 OT를 진행하시면 됩니다!
    첫 모임에 여러분들이 선정하신 러너분들과 함께하시면서, 서로 생각한 모임의 방향이 맞는지 확인합니다. 얼라인 하는 과정에서 거리를 좁히지 못하거나, 절대로 참여가 불가능한 일정이라면, 러너분들이 활동을 "취소"할 수 있습니다. 그렇게 되면 빈 자리가 생기게되고 선정되지 못한 러너들 중에 추가 선정하실 수 있는 기간이 생깁니다!
    (기존에는 이 과정이 없어서, 선정된 리너가 첫 모임 이후부터 빠르게 이탈하는 경우들이 있었습니다.)
    
    즉, 팀별 조정기간이 끝난 후에 최종적인 "탈락" 발표가 이루어집니다.
    
    팀별 조정 기간에 신청/선정을 취소하고 싶은 경우, 김찬란에게 DM을 보내주시면 처리!
    
    (권한 등 빠른 처리가 필요할 것으로 판단)
    

### 매 주 routine

- 7시~9시 (optional): ~~모각코~~ 그냥 놀아요
    - 되는 사람만 모이기
    - 오프라인에서 모입니다. (현재 공덕)
    - 소통하는 시간
- 9시 ~ 10시 : 발표
    1. 어떤 **문제**를, 어떤 **과정**으로 해결하는지 그룹에 소개하기
    2. 1을 직접 만들어보기
    3. **지난주 질문 해결**  
- 디스코드를 적극 활용해주세요.
    - 뭐 build했다 이런거 올리고, (build on public)
    - 읽은 분들은 박수쳐주자

- 질의응답

그리고 제가 만들던거 (로드맵 요약기)

