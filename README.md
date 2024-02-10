## Introduction:
- This notebook presents a movie recommender system built using the TMDB 5000 Movie Dataset. The system utilizes natural language processing techniques to analyze movie descriptions, genres, keywords, cast, and crew information. By computing similarity scores between movies based on these features, the system recommends similar movies to users based on their input. The implementation involves data preprocessing, text processing, vectorization, and cosine similarity computation. The notebook also demonstrates how to save the necessary data for deployment, including movie dataframes and similarity matrices.


```python
#import all necessary libraries
import numpy as np
import pandas as pd
import ast
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```



```python
#import data 
movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')
```

```python
movies.head()
```

```python
credits.head()
```
<img width="747" alt="Screenshot 2024-02-10 193255" src="https://github.com/Pratiksha022002/Movie_recommender_system/assets/99002937/bdec0217-cdae-4b11-8350-b71b57685176">

- Merge movies and credits dataset on 'title' column
```python
movies = movies.merge(credits,on='title')
```

```python
# Including only relevant columns
movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.head()
```

### Data Transformation

- ast.literal_eval converts string to list
- convert dictionary into a list with required elements
```python
def convert(obj):
    L=[ ]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L
   
```


```python
movies['genres']=movies['genres'].apply(convert)
```

```python
 movies['keywords']=movies['keywords'].apply(convert)
```

-  Function to extract top 3 actors from cast list
```python

def convert3(obj):
    L=[ ]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            L.append(i['name'])
        else:
            break
    return L
```


```python
movies['cast']=movies['cast'].apply(convert3)
```
- Function to fetch only director's name from crew

```python
def fetch_director(obj):
    L=[ ]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L
```

```python
movies['crew']=movies['crew'].apply(fetch_director)
```

- convert string to list
```python
movies['overview']=movies['overview'].apply(lambda x:x.split()) 
```


```python
#concatenate overview,genres,keyword,cast and crew to form a new column tags
movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']
```


```python
movies.head()
```
<img width="833" alt="Screenshot 2024-02-10 193357" src="https://github.com/Pratiksha022002/Movie_recommender_system/assets/99002937/bdde8ab7-abea-412e-b212-0df63d683c98">

- Histogram for genres


```python
all_genres = [genre for sublist in movies['genres'] for genre in sublist]

# Plot histogram for genres
plt.figure(figsize=(8, 3))
sns.histplot(all_genres, color='skyblue', discrete=True)
plt.xlabel('Genre')
plt.ylabel('Frequency')
plt.title('Distribution of Genres')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()
```
<img width="583" alt="Screenshot 2024-02-10 193715" src="https://github.com/Pratiksha022002/Movie_recommender_system/assets/99002937/476df3f5-11cd-43b4-b058-74bd87157912">

```python
#in new data frame only consider three columns,movie_id,title,tags 
new_df=movies[['movie_id','title','tags']]
```


## Text Processing


- Stemming is a natural language processing technique that is used to reduce words to their base form, also known as the root form.


```python
ps=PorterStemmer()
def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
```


```python
#apply stemming on tags so no repeated words occur
new_df['tags']=new_df['tags'].apply(stem)
```


```python
#convert strig to lowercase
new_df['tags']=new_df['tags'].apply(lambda x:x.lower())
```


```python
tags_text = ' '.join(new_df['tags'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(tags_text)

plt.figure(figsize=(6, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Movie Tags')
plt.show()
```
<img width="403" alt="Screenshot 2024-02-10 193733" src="https://github.com/Pratiksha022002/Movie_recommender_system/assets/99002937/c2eccdb4-94f2-468d-99c3-0b792df0abf1">


## Model Building
- Concatenate all tags and convert text to vector 
get 5000 most occuring words in a large text
don't consider stop words.

```python
cv=CountVectorizer(max_features=5000,stop_words='english')
```

```python
vectors=cv.fit_transform(new_df['tags']).toarray()
```


- Cosine Similarity :
cosine_similarity calulate theta between two vectors


```python
similarity=cosine_similarity(vectors)
```


```python
plt.figure(figsize=(8, 3))
plt.scatter(range(len(similarity)), similarity[0], color='orange', alpha=0.6)
plt.xlabel('Movie Index')
plt.ylabel('Similarity Score')
plt.title('Similarity Scores with Movie 0')
plt.show()
```
<img width="527" alt="Screenshot 2024-02-10 193746" src="https://github.com/Pratiksha022002/Movie_recommender_system/assets/99002937/1a512877-4432-4350-a24d-8f5c48422169">

- We want to find 5 similar movies based on max cosine_similarity


```python
def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index]            
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6] 
    for i in movies_list:
        a=i[0]                                   
        print(new_df.iloc[a].title)
    
```


```python
def plot_recommendations(movie):
    plt.figure(figsize=(5, 3))
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies = [new_df.iloc[movie[0]].title for movie in movies_list]
    sns.barplot(x=[movie[1] for movie in movies_list], y=recommended_movies, palette='rocket')
    plt.title(f"Recommendations for '{movie}'")
    plt.xlabel('Similarity Score')
    plt.ylabel('Movies')
    plt.show()
```


```python
# Plot recommendations for a specific movie
plot_recommendations('Avatar')
```
<img width="503" alt="Screenshot 2024-02-10 193757" src="https://github.com/Pratiksha022002/Movie_recommender_system/assets/99002937/cd2fb236-c4bb-4e81-a048-51e837b406df">
