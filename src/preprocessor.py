import time
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import *
import numpy as np 
import re
from nltk.stem import *
import nltk
from predictor import *


#runs dataframe through all functions listed below
def clean(data):
	clean_tweets = []
	for tweet in data.tweet:
		noiseless = remove_noise(tweet)
		stemmed = to_stems(noiseless,False)
		clean_tweets.append(stemmed)
	data.tweet = clean_tweets


#takes in a string and uses regex to filter out unnecessary characters/symbols
def remove_noise(words:str) -> str:
	#remove special characters
	words = re.sub(r'[^a-zA-Z]', ' ', words)
	#remove single characters
	words = re.sub(r'\s+[a-zA-Z]\s+', ' ', words)
	#Substitute multiple spaces with single space
	words = re.sub(r'\s+', ' ', words, flags=re.I)
	return words

#lemmatizes and stemms each string passed through it, returns them as a string
def to_stems(words:str, stopword:bool) -> str:
	lemmatizer = WordNetLemmatizer()
	words =  [lemmatizer.lemmatize(word) for word in words.split()]
	#print((words))
	stemmer = SnowballStemmer('english',ignore_stopwords = stopword)
	words = [stemmer.stem(word) for word in words]
	ret_string = ''
	for word in words:
		ret_string += word + ' '
	ret_string = ret_string[:-1]
	#print(words)
	return ret_string

#creates a bag of words with tfidf scores for each word from dataframe
def bag(data):
	tweets = []
	for record in data:
		tweets.append(record)	#collect the tweets into an array of tweets
	vectorizer = CountVectorizer()
	word_bag = vectorizer.fit_transform(tweets)		#type scipy.sparse.csr.csr_matrix (compressed sparse row matrix)
	tfidf_transformer = TfidfTransformer()
	tfidf = tfidf_transformer.fit_transform(word_bag)		#tfidf weighted word bag
	return tfidf

if __name__ == '__main__':
	main()