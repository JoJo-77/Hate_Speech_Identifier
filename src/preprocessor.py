import time
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import *
import numpy as np 
import re
from nltk.stem import *
import nltk
from predictor import *

#things to do:
# thread out everything
# create main function that ties all 3 parts together
# collect data from each part 
# finish report


def main():
	s = time.time()
	make_clean = True
	test = None
	train = None
	tfidf_bag = None
	if make_clean:
		train = pd.read_csv("train.csv")
		test = pd.read_csv("test.csv")
		clean(train)
		clean(test)
		train.to_csv("clean_train.csv", index=False, encoding='utf8')
		test.to_csv("clean_test.csv", index=False, encoding='utf8')
	else:
		train = pd.read_csv("clean_train.csv")
		test = pd.read_csv("clean_test.csv")
	print("Data processed")
	tfidf_bag = bag(train)	#rows = # of records 31,962		columns = dict size 37,543
	#uncomment line below to run predictions. takes about 540 seconds
	predict(tfidf_bag, train.label)

	e = time.time()
	print("total runtime = " + str(e - s) + " seconds")
	tfidf_bag = bag(train)




def clean(data):
	clean_tweets = []
	for tweet in data.tweet:
		noiseless = remove_noise(tweet)
		stemmed = to_stems(noiseless,False)
		clean_tweets.append(stemmed)
	data.tweet = clean_tweets


#preprocessing steps:
#all lower case -> already done
#stemming -> Snowball
#stopword removal -> in stemmer
#normalization (ex: gud -> good / goooood -> good) -> we'll see if we need, i dont think we will
#noise removal (remove symbols and numbers) -> use regex include hashtags and apostrophes

def remove_noise(words:str) -> str:
	#remove single characters
	words = re.sub(r'\s+[a-zA-Z]\s+', ' ', words)
	#Substitute multiple spaces with single space
	words = re.sub(r'\s+', ' ', words, flags=re.I)
	#remove special characters except ' and #
	words = re.sub(r'[^a-zA-Z\']', ' ', words)
	return words

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


def bag(data):
	tweets = []
	for record in data.tweet:
		tweets.append(" ".join(record))
	vectorizer = CountVectorizer()
	word_bag = vectorizer.fit_transform(tweets)		#type scipy.sparse.csr.csr_matrix (compressed sparse row matrix)
	tfidf_transformer = TfidfTransformer()
	tfidf = tfidf_transformer.fit_transform(word_bag)		#tfidf weighted word bag
	return tfidf

if __name__ == '__main__':
	main()