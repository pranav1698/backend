import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
from nltk import FreqDist
import warnings 
from nltk.stem.porter import  *
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import brown
from nltk.collocations import *
from sklearn.feature_extraction.text import TfidfVectorizer

#stop_words = set(stopwords.words('english'))



train = pd.read_csv('./output.csv')
#print (train['text'])
tweets=train['text']



train['text'] = train['text'].str.replace("[^a-zA-Z#]", " ") 

#Removing Short Words

train['text'] = train['text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))



#tokenization
tokenized_tweet = train['text'].apply(lambda x: x.split())


#print (tokenized_tweet)
#Stemming is done in this part
stemmer = PorterStemmer()

#tokenized_tweet = tokenized_tweet.apply(lambda x:[stemmer.stem(i) for i in x])
#print(tokenized_tweet)
new_tweet=tokenized_tweet
#print(new_tweet)


#join all tokenized and stemmed tokens together
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i]=' '.join(tokenized_tweet[i])
train['text'] = tokenized_tweet
#print(train['text'].head())

# removing https:// and URLs


   
    

#creating a .csv file storing values




#print(tweets)
def hashtag_extract(x):
    hashtag=[]
    for i in x:
        ht =re.findall(r"(\w+)", i)
        hashtag.append(ht)
        
    return hashtag

# extracting hashtags from tweets
HT_regular = hashtag_extract(train['text'])

cleaned_tweet=HT_regular
#print(cleaned_tweet)



HT_regular = sum(HT_regular,[])


#Plotting non racist hastag
a = nltk.FreqDist(HT_regular)
a.plot(30)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
# selecting top 10 most frequent hashtags     
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(15,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')

plt.savefig('/home/rrahul/seabornDesktop/figure_1.png',dpi=100)
plt.show()

#Location
location=train['id']
print(location)