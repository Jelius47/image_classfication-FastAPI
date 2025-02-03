import pickle #For serialization and export it to be used in our API
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


X,y = fetch_openml('mnist_784',version=1,return_X_y=True) # meaning 28*28 pixels

X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2) # 20% test data

classifier = RandomForestClassifier(n_jobs=-1)#n_jobs will cause the usage of all cpu core to speed up

# Model fitting 
classifier.fit(X_train,y_train)

# Evaluation of the model
print(classifier.score(X_test,y_test))

# Saving the model
with open('mnist_model_image_classifier.pkl','wb')as f:# 'wb' signifies writing bites mode
    pickle.dump(classifier,f)