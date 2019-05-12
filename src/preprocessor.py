import pandas as pd 
import sklearn 
import re
from nltk.stem import *
import threading


def main():
	make_clean = False
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
	regex = re.compile('[^a-zA-Z#\']')
	return regex.sub(' ',words)

def to_stems(words:str, stopword:bool) -> list:
	stemmer = SnowballStemmer('english',ignore_stopwords = stopword)
	stems = []
	for word in words.split():
		stemmer.stem(word)
		stems.append(word)
	return stems

if __name__ == '__main__':
	main()