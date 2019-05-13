import predictor
from preprocessor import *
import classifier

def main():
	s = time.time()
	#store data into dataframes
	train = pd.read_csv("train.csv")
	test = pd.read_csv("test.csv")
	#preprocess data
	clean(train)
	clean(test)

	print("Data processed after " + str(time.time() - s) + " sec")


#-----------Predictor-------------------------------
	tfidf_bag = bag(train.tweet)
	predictor.predict(tfidf_bag, train.label)
#---------------------------------------------------

#-----------Clustering------------------------------
	#create deep copy of training data
	hateful = train.copy(deep = True)
	#separate out hateful tweets
	classifier.get_hateful(hateful)
	#train classifier
	kmeans_model, vectorizer = classifier.train(hateful,False)
	print("Clusters Found after " + str(time.time() - s) + " sec")
#------------------------------------------------

	print("total runtime: " + str(time.time()-s) + " sec")

if __name__ == '__main__':
	main()