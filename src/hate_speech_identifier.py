import predictor
from preprocessor import *
import classifier
import threading

def main():
	s = time.time()
	train = pd.read_csv("train.csv")
	test = pd.read_csv("test.csv")
	hateful = classifier.get_hateful(train)
	train_thread = threading.Thread(target = clean, args = (train))
	test_thread = threading.Thread(target = clean, args = (test))
	train_thread.start()
	test_thread.start()
	test_thread.join()
	test_thread.join()
	#before threading: 107 sec
	#after threading: 94 sec -> 13 second improvement
	print("Data processed after " + str(time.time() - s) + " sec")
<<<<<<< HEAD
	tfidf_bag = bag([x[1][2] for x in train.iterrows()])
=======
	train = pd.read_csv("train.csv")
	clean(train)
	tfidf_bag = bag(train.tweet)
	predict(tfidf_bag, train.label)
>>>>>>> 6885afadbdc5f615b26443f65971afaf1970dd7d

#-----------Insert Predictor Here-------------------
#Needs to return a df of hateful tweets from test, 
#would be cool if you could add to the hateful dataframe above
#Note: usage of all objects/methods in predictor need to follow
#predictor.foo() format to prevent errors (we both have a predict function)
#---------------------------------------------------

#-----------Clustering------------------------------
	kmeans_model, vectorizer = classifier.train(hateful,False)
	print("Clusters Found after " + str(time.time() - s) + " sec")


if __name__ == '__main__':
	main()