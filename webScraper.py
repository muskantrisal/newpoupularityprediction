
import nltk
import datetime
from newspaper import Article
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize  
from textblob import TextBlob
from nltk.tokenize import sent_tokenize, word_tokenize 





nltk.download('stopwords')



#title lenght , content lenght , rate of unique token , rate of unique stopwords, number of images , average token length , days , isWeekend 
#global subjectivity , global polarity , title subjectivity , title polarity



#number of tokens in tittle
def title_lenght(article):
    title = article.title
    title = title.split()
    lenght = len(title)
    return lenght

#number of tokens in content
def text_lenght(article):
    text = article.text
    text = text.split()
    lenght = len(text)
    return lenght  

#number of unique tokens
def unique_tokens(article):
    d = {}
    text = article.text
    text = text.split()
    for i in text:
        d[i] = d.get(i,0) + 1 
    unique_token_rate = len(d)/len(text)
    return unique_token_rate   

#rate of unique stop worlds
stop_words = set(stopwords.words('english')) 
def unique_stop(article):
    text = article.text
    word_tokens = word_tokenize(text) 
    stop = [w for w in word_tokens if w in stop_words] 
    stop_unique = {}
    for i in stop:
        stop_unique[i] = stop_unique.get(i,0) + 1
    return len(stop_unique)/len(stop)

#counting number of images 
def number_images(article):
    return len(article.images)  

#finding average token length
def average_token_length(article):
    text = article.text
    tokens = word_tokenize(text)
    return len(text)/len(tokens)  

#day
def day_array(article):
    week_dict = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}
    date = article.publish_date
    if date is not None:
      day = date.strftime("%A")
    else:
      day = 'Sunday'
    week_list = 7*[0]
    week_list[week_dict[day]] = 1 
    return week_list

#isweekwnd
def is_weekend(week):
    if week[5] == 1 or week[6] == 1:
        return 1
    else:
        return 0  

#global subjectivity of text
def text_subjectivity(article):
    text = article.text
    obj_text = TextBlob(text)
    subjectivity_text = obj_text.sentiment.subjectivity
    return subjectivity_text

#global polarity of text
def text_polarity(article):
    text = article.text
    obj_text = TextBlob(text)
    sentiment_text = obj_text.sentiment.polarity
    return sentiment_text

#title subjectivity
def title_subjectivity(article):
    title = article.title
    obj_title = TextBlob(title)
    subjectivity_title = obj_title.sentiment.subjectivity
    return subjectivity_title

#title polarity
def title_polarity(article):
    title = article.title
    obj_title = TextBlob(title)
    polarity_title = obj_title.sentiment.polarity
    return polarity_title  


def required_details(url):
  #declaring list to store the parameters
    data = []

    article = Article(url)
    article.download()
    article.parse()
    nltk.download('punkt')
    article.nlp()
    stop_words = set(stopwords.words('english')) 
    nltk.download('stopwords')

  #getting number of tokens in title
    n_tokens_title = title_lenght(article)
    data.append(n_tokens_title)

  #getting number of tokens in content
    n_tokens_content = text_lenght(article)
    data.append(n_tokens_content)

  #getting rate of unique tokens
    n_unique_tokens = unique_tokens(article)
    data.append(n_unique_tokens)

  #getting number of unique stopwords
    n_non_stop_unique_tokens = unique_stop(article)
    data.append(n_non_stop_unique_tokens)

  #counting number of images
    num_imgs = number_images(article)
    data.append(num_imgs)

  #average length of token
    average_token = average_token_length(article)
    data.append(average_token)

  #day
    day = day_array(article)
    for i in day:
        data.append(i)

  #isweekend
    week = day_array(article)
    weekend = is_weekend(week)
    data.append(weekend)   

  #calculating global subjectiity
    global_subjectivity = text_subjectivity(article)
    data.append(global_subjectivity)

  #calculating global sentimental polarity
    global_sentiment_polarity = text_polarity(article)
    data.append(global_sentiment_polarity)

  #calculating title subjectivity
    subjectivity = title_subjectivity(article)
    data.append(subjectivity)

  #calculating title polarity
    title_sentiment_polarity = title_polarity(article)
    data.append(title_sentiment_polarity)

    return data
