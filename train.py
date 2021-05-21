import numpy as np
import sklearn.svm as sk
# define data
featureTr = [[1, 3], [2, 2], [1, 1], [3, 1], [4, 5], [5, 4]]
labelTr = [1, 1, 1, 1, 2, 2]
featureTs = [4, 4]
# create SVM Object
clf = sk.SVC()
# train SVM model
clf.fit(featureTr, labelTr)
# classify by using SVM
out = clf.predict([featureTs])
print('Answer is ' + str(out))
