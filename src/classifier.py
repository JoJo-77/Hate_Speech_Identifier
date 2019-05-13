from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from preprocessor import *


#trains the kmeans classifier and returns model ready for predicting
def train(file):
	#get csv into dataframe
	train = pd.read_csv(file)
	#clear out non-hateful tweets
	get_hateful(train)
	#clean hateful tweets 
	clean(train)
	#init vectorizer
	vectorizer = TfidfVectorizer(stop_words='english')
	#fit vectorizer with all tweets
	X = vectorizer.fit_transform([x[1][2] for x in train.iterrows()])
	#save all feature names
	terms = vectorizer.get_feature_names()
	#initialize and fit model on all tfidf tweet scores
	cluster_num = 2
	model = KMeans(n_clusters=cluster_num, init='k-means++', max_iter=400, n_init=1)
	model.fit(X)
	#print out top terms in each cluster
	print("Top Terms Per Cluster:")
	ordered_centroids = model.cluster_centers_.argsort()[:,::-1]
	terms = vectorizer.get_feature_names()
	for i in range(cluster_num):
		print("Cluster %d:" % i),
		for ind in ordered_centroids[i, :10]:
			print(' %s' % terms[ind])
	return model

#accepts a model and a string of text to be predicted, runs string through model 
#returns predicted string
def predict(model,string)
	print("Prediction")
	Y = vectorizer.transform([remove_noise(to_stems(string))])
	prediction = model.predict(Y)
	print(prediction)
	return prediction

def get_hateful(df):
	index = 0
	for row in df.iterrows():
		#check if not hateful and drop row if so
		if row[1][1] == 0:
			df.drop([index],inplace = True)
		index += 1
	print(df.shape)

if __name__ == '__main__':
	main()


