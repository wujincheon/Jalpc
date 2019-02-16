---
layout: post
title: "From News Crawling to TF-IDF"
date: 2019-01-28 
desc: "From News Crawling to TF-IDF"
keywords: "HTML, Crawling, TF-IDF"
categories: [Natural language processing]
tags: [HTML, Crawling, TF-IDF]
icon: icon-html
---

이번 포스팅에서는 뉴스 기사를 수집하는 html 코드를 통해 그 과정을 살펴보겠습니다. 그리고 수집한 기사를 바탕으로 TF-IDF(Term Frequency - Inverse Document Frequency)를 적용하여 문서 내 단어들의 중요도에 대한 정보를 얻어보겠습니다.  

---

## 1. News Crawling  
##### 크롤링할 page list 생성  
![](https://i.imgur.com/O0nwfLs.png?1)  
다음 뉴스에서 경제 관련 뉴스를 수집하고 싶다고 하겠습니다. 전체기사에서 경제 영역으로 들어가면 경제 뉴스들이 순서대로 나타나는 것을 볼 수 있습니다. 그리고 원하는 날짜를 선택하면 해당 날짜에 나왔던 기사들을 볼 수 있습니다. 위 그림처럼 경제를 선택하고 날짜를 선택한 인터넷 창입니다. 위에 인터넷 주소창을 보면 economic을 볼 수 있고 'regDate=' 를 통해 원하는 날짜를 입력하면 바로 창을 이동할 수 있음을 알 수 있습니다. 그리고 크롤링하고 싶은 기사를 눌러보겠습니다.  

![](https://i.imgur.com/nvJYKjA.png?1)  
그랬더니 인터넷 주소가 해당 기사가 올라온 시간에 맞춰서 변한 것을 볼 수 있었습니다. 원래는 여기서 나타나는 주소를 이용해서 내용을 받으면 됩니다. 그런데 저기에 발생하는 시간을 일일히 다 알 수 없었기에 저는 이 창이 아닌, 처음 그림에서 나타나는 창에서 바로 크롤링을 시도했습니다.  

원하는 날짜의 기사 목록들이 나타나는 창에서 어떻게 각각의 기사의 주소를 얻을 수 있을까요?  
바로 html 코드상에서 나타나는 각 기사의 주소를 크롤링하면 되는 것입니다. 그럼 html 코드를 페이지 소스 보기를 통해 살펴보겠습니다.  

![](https://i.imgur.com/PG3IDRA.png?1)  
각 기사를 나타내는 제목 앞에 보면 a href = "http~" 의 형식으로 링크가 걸려있는 것을 확인할 수 있습니다. 그래서 저는 BeautifulSoup 패키지를 이용해서 원하는 날짜에 대한 페이지에서 각각의 기사 제목들을 클릭하면 나타나는 기사들의 인터넷 주소들을 얻어냈습니다. 추가적으로 해당 날짜마다 창에서는 15개씩의 기사밖에 보여주지 않기 때문에, 페이지를 20여 페이지까지 늘려서 총 300여개의 기사 링크를 얻었습니다. 그리고 날짜도 변경시켜가며 일주일치, 총 2000여개의 기사 링크를 먼저 리스트로 저장하는 작업을 진행했습니다. 아래 코드를 참고하시면 됩니다.  

```python
import urllib.request
from bs4 import BeautifulSoup
url_1='https://media.daum.net/breakingnews/economic?regDate='
date=range(20131005,20131012)#1

url_2='&page='
page= range(1,20)

url_list=[]
for i in date:
    for j in page:
        mainurl=url_1 +str(i)+ url_2 + str(j) 
        source_code_from_URL = urllib.request.urlopen(mainurl, context=context)
        soup = BeautifulSoup(source_code_from_URL, 'lxml', from_encoding='utf-8')
        list_news = soup.find('ul', attrs={'class':'list_news2'})
        list_news_li = list_news.find_all('li')
        for item in list_news_li:
            link_txt = item.find('a', attrs={'class':'link_txt'})
            url_list.append(link_txt.get('href')) #위에서 말한 링크가 나타나는 부분을 찾는 코드입니다.
```

##### 기사 제목 및 내용 크롤링 진행  
![](https://i.imgur.com/stO7rVk.png)  
이렇게 위에서 얻은 기사들에 대한 링크를 바탕으로 아래의 코드를 이용해서 크롤링을 진행했습니다. 기사에 대한 제목과 내용을 합쳐서, 그 내용에 대해 한국어 명사를 끌어내는 함수를 통해 기사에서 나타나는 단어들을 수집합니다. 원래의 기사만을 추출하려면 굳이 kkma를 쓰지 않아도 되지만 뒤에서 적용할 TF-IDF는 단어가 들어가야 하기 때문에 단어를 따로 저장했습니다.
```python
from newspaper import Article
from konlpy.tag import Kkma, Twitter, Komoran

kkma=Kkma()


mydoclist_kkma=[]


for url in url_list:
    article=Article(url, language='ko')
    article.download()
    article.parse()
    hoho=article.title+article.text
    
    kkma_nouns = ' '.join(kkma.nouns(hoho))
    mydoclist_kkma.append(kkma_nouns)


```



---


## 2. TF-IDF   
TF-IDF 는 Term Frequency 즉, 단어가 나타나는 빈도와 Inverse Document Frequency, 하나의 단어가 몇 개의 문서에서 나타나는지에 대한 빈도의 역수 값을 곱해주는 개념입니다. 즉 이 값이 의미하는 바를 생각해보면, TF가 클수록 그리고 DF값이 작을수록 중요도가 크다고 판단되는 것입니다. TF값이 크다는 것은 해당 문서에서 그 단어가 빈번하게 나타난다는 것으로, 해당 기사에서 주 내용을 담당하고 있는 중요 단어라고 이해할 수 있습니다. 그리고 DF값이 작다는 것은 여러 문서에서 나타나는 단어가 아니고, 해당 기사에서만 나타나는 Unique한 단어라는 의미입니다. 정리해보자면, 해당 날짜에 TF-IDF 값이 높은 단어라는 것은, 수많은 기사에서 나타나지 않고 하나의 기사에서 중요하게 다뤄지는 단어로, 해당 날짜에 발생한 특별한 이슈라고 해석해볼 수 있다는 것입니다.  
이 값들은 다양하게 활용될 수도 있는데, DF값이 높다는 것은 해당 날짜에서 대부분의 문서에서 나타나는 단어로, 해당 날짜의 일들을 대표할 수 있는 무언가를 얻어낼 수도 있다는 것입니다. 해당 패키지에서 DF값이나 TF값을 자유롭게 얻을 수 있고 변형시킬 수 있으므로 다양하게 쓰일 수 있을 것 같습니다.  
이제껏 설명한 TF-IDF의 코드는 아래와 같이 매우 간단하게 얻어낼 수 있습니다. 
  
```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(min_df=1)
tfidf_vectorizer.fit(mydoclist_kkma)
tfidf_matrix_kkma =tfidf_vectorizer.transform(mydoclist_kkma)
```

---

## 3. 단어 분포 유사도 
이렇게 얻어낸 단어 분포를 이용해서 여러 가지 일들을 할 수 있는데, 단어들을 시각화해서 워드클라우드를 만들어 볼 수도 있고, 날짜별로 얻어낸 단어 분포간의 차이를 계산해볼 수도 있습니다.  
![](https://i.imgur.com/JmHR7t0.png?1)  
두 날짜에 해당하는 TF-IDF 값을 기준으로 한 단어 분포를 어떻게 비교할 수 있을까요? 먼저 두 날짜에서 나타나는 단어가 다르기 때문에, 두 날짜에서 나타나는 전체 단어를 기준으로 벡터를 다시 만들어줄 필요가 있습니다.  

![](https://i.imgur.com/NzivaVv.png)  
이렇게 다시 만들어낸 벡터를 기준으로 코사인 유사도 값을 구해볼 수 있습니다. 그리고 각 단어를 나타내는 TF-IDF값을 다양하게 바꿔보면서 실험을 해볼수도 있을것 같습니다. 또한 날짜별로 TF-IDF 값의 합을 1이 되게끔 비율로 나타낸 다음에 KL-Divergence와 JS-Divergence와 같은 수치를 통해서 두 날짜의 단어 분포 유사도를 측정해볼 수 있습니다.



---


## 4. 결론  
뉴스 기사를 크롤링할 때는 시간이 매우 오래걸리기 때문에, 작은 규모로 테스트를 해보고 진행하는 것이 좋은 것 같습니다. 그리고 크롤링에 대한 오류때문에, 각 기사를 수집할때마다 중간중간에 타임슬립을 넣어주는 것이 좋습니다. 그리고 수집하고자 하는 페이지마다 html 코드도 다르고 형식도 매우 다르기 때문에, 각 페이지의 구조나 코드를 완전히 이해하고 가져다 써야 합니다.  
그리고 그렇게 수집된 정보를 TF-IDF에 적용해서 단순한 유사도를 비교하는 작업을 거쳤지만, 사실 실제로는 이것이 유의미한 작업이 되려면 상당한 전처리나 중간 작업이 필요합니다. 필요없는 단어들을 쳐내는 과정들이나 어느 정도까지 단어들을 반영할 것인지나, 어떤 값을 이용해서 정렬하고 합치고 이용할 것인지, 그리고 그 검증은 어떤 것들과 비교해서 타당성을 결론지을 것인지 등에 대해 생각해야 합니다.  
데이터마다 특성이 다르고, 하고자하는 주제마다 특성이 다르기 때문에, 그 데이터에 맞는 처리 방법과 검증 방법이 다양하게 시도되어야 할 것이고, 이 포스팅에서 나타나지 않은 다양한 방법들도 존재하기 때문에 열심히 공부해야 한다고 새삼 느꼈습니다.

---

## 다음 포스팅에는..

Object Detection과 관련된 개념들과 논문을 살펴보겠습니다.
