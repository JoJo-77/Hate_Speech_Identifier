import predictor
from preprocessor import *
import classifier
import threading

def main():
	s = time.time()
	train = pd.read_csv("train.csv")
	test = pd.read_csv("test.csv")
	clean(train)
	clean(test)

	print("Data processed after " + str(time.time() - s) + " sec")


#-----------Predictor-------------------------------
	tfidf_bag = bag(train.tweet)
	predictor.predict(tfidf_bag, train.label)
#---------------------------------------------------

#-----------Clustering------------------------------
	hateful = train.copy(deep = True)
	get_hateful(hateful)
	kmeans_model, vectorizer = classifier.train(hateful,False)
	print("Clusters Found after " + str(time.time() - s) + " sec")

#------------------------------------------------
if __name__ == '__main__':
	main()