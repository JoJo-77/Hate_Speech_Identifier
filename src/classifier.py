from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from preprocessor import *



def main():
	train = pd.read_csv("train.csv")

	#clear out non-hateful tweets
	get_hateful(train)
	#clean hateful tweets
	clean(train)

	#tfidf_bag = bag(train)
	print(train.shape)

	get_hateful(train)

	vectorizer = TfidfVectorizer(stop_words='english')
	X = vectorizer.fit_transform([x[1][2] for x in train.iterrows()])
	terms = vectorizer.get_feature_names()

	model = KMeans(n_clusters=2, init='k-means++', max_iter=400, n_init=1)
	model.fit(X)
	print("Top Terms Per Cluster:")
	order_centroids = model.cluster_centers_.argsort()[:,::-1]
	terms = vectorizer.get_feature_names()
	for i in range(3):
		print("Cluster %d:" % i),
		for ind in order_centroids[i, :10]:
			print(' %s' % terms[ind])

	print("\n")
	print("Prediction")
	Y = vectorizer.transform([remove_noise(to_stems("hate blacks, black people suck hate,mexicans, racists, genocide",True))])
	prediction = model.predict(Y)
	print(prediction)

	Y = vectorizer.transform([remove_noise(to_stems("women suck, vaginas are bad, slut,  cunt, whore ,nazi, sjw, libtard, femanism, femenazi,",True))])
	prediction = model.predict(Y)
	print(prediction)

	Y = vectorizer.transform([remove_noise(to_stems("politics, milo, go trump, bad hillary, maga, make america great again, obama",True))])
	prediction = model.predict(Y)
	print(prediction)

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


