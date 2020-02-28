import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection, naive_bayes, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Reads in CSV
audio_features_df = pd.read_csv('audio_features.csv')

from sklearn import decomposition

X = audio_features_df.iloc[:,:-1]
y = audio_features_df['class']

# PCA decomposition for dimensionality reduction
pca = decomposition.PCA(n_components=10,random_state=0)
pca.fit(X)
X = pca.transform(X)

# Train-test split
X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size=0.1, random_state = 0)

# Label encoding for the classes
Encoder = LabelEncoder()
y_train = Encoder.fit_transform(y_train)
y_test = Encoder.transform(y_test)

# Train a random forest classifier
print('Training random forest model')
rf = RandomForestClassifier()
rf.fit(X_train,y_train)

# Cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = rf, X = X, y = y, cv = 10,n_jobs=-1)

print('accuracies mean',accuracies.mean())
print('accuracies std',accuracies.std())
print('Model training complete')

# Saves model to disk
import pickle
model_filename = 'model.pkl'
pickle.dump([rf,pca,Encoder], open(model_filename, 'wb'))
print('Model saved to',model_filename)

# Visualize the results of 2D PCA
pca_2d = decomposition.PCA(n_components=2,random_state=0)
pca_2d.fit(audio_features_df.iloc[:,:-1].values)

# Saves 2D PCA model to file
pca_2d_filename = 'PCA_2D.pkl'
pickle.dump(pca_2d, open(pca_2d_filename, 'wb'))
print('2D PCA model saved to',pca_2d_filename)