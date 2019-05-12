import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score

def predict(bag, labels):
    print("Beginning predictions...")

    print("Bag of words shape: {}".format(bag.shape))

    #feature selection:     reduces number of attributes by about 80%
    sel = VarianceThreshold(threshold=(.00002 * (1 - .00002)))
    new_bag = sel.fit_transform(bag)
    print("Shape after feature selection: {}".format(new_bag.shape))

    #split data
    X_train, X_test, y_train, y_test = train_test_split(new_bag, labels, test_size=0.2, random_state=1, stratify=labels)

    #F1: 0.57   runtime: 10 seconds
    decision_tree(X_train, X_test, y_train, y_test)

    #F1: 0.70   runtime: 530 seconds
    #With FS: F1: 0.62  runtime: 401 seconds
    #neural_network(X_train, X_test, y_train, y_test)

    X_train = X_train[:5000]
    X_test = X_test[:1000]
    y_train = y_train[:5000]
    y_test = y_test[:1000]

    print("shape of Xtrain, Xtest:  {}, {}".format(X_train.shape, X_test.shape))

    #naive bayes and knn cause memory errors
    naive_bayes(X_train, X_test, y_train, y_test)

    k_nearest(X_train, X_test, y_train, y_test)


def decision_tree(X_train, X_test, y_train, y_test):
    s = time.time()
    d_tree = tree.DecisionTreeClassifier()
    d_tree.fit(X_train, y_train)
    preds = d_tree.predict(X_test)
    f1 = f1_score(y_test, preds)
    print("\nDecision tree F1 score: {}".format(f1))
    e = time.time()
    print("Decision tree runtime: " + str(e - s) + " seconds\n")

def naive_bayes(X_train, X_test, y_train, y_test):
    s = time.time()
    bayes = GaussianNB()
    bayes.fit(X_train.todense(), y_train)
    preds = bayes.predict(X_test.todense())
    f1 = f1_score(y_test, preds)
    print("\nNaive Bayes F1 score: {}".format(f1))
    e = time.time()
    print("Naive Bayes runtime: " + str(e - s) + " seconds\n")


def neural_network(X_train, X_test, y_train, y_test):
    s = time.time()
    neural = MLPClassifier()
    neural.fit(X_train, y_train)
    preds = neural.predict(X_test)
    f1 = f1_score(y_test, preds)
    print("\nNeural network F1 score: {}".format(f1))
    e = time.time()
    print("Neural network runtime: " + str(e - s) + " seconds\n")

def k_nearest(X_train, X_test, y_train, y_test):
    s = time.time()
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    f1 = f1_score(y_test, preds)
    print("\nKNN F1 score: {}".format(f1))
    e = time.time()
    print("KNN runtime: " + str(e - s) + " seconds\n")

