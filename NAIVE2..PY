import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score

msg=pd.read_csv('./8)document.csv',names=['message','label'])

msg['labelnum']=msg['label'].map({'pos':1,'neg':0})


X=msg['message']
y=msg['labelnum']


Xtrain,Xtest,ytrain,ytest=train_test_split(X,y)

count_v=CountVectorizer()
Xtrain_dm=count_v.fit_transform(Xtrain)
Xtest_dm=count_v.transform(Xtest)


clf=MultinomialNB()
clf.fit(Xtrain_dm,ytrain)
pred=clf.predict(Xtest_dm)

print(pred.shape)
print(Xtest.shape)


for doc,p in zip(Xtest,pred):
    p='pos' if p==1 else 'neg'
    print("%s -> %s"%(doc,p))

print(accuracy_score(ytest,pred))
print(recall_score(ytest,pred))
print(confusion_matrix(ytest,pred))
print(precision_score(ytest,pred))
