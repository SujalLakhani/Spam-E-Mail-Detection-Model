import pandas as pd
import matplotlib.pyplot as plt

#Reading the Dataset SpamOrNot.csv
df=pd.read_csv('./SpamOrNot.csv')

#Details of the Dataset
print(df.shape)
print(df.size)
print(df.ndim)
print(df.head())
print(df.tail())
df.info()

#Independent variables
x=df.iloc[:,:-1].values

#Dependent variable
y=df.iloc[:,-1].values

#Visualization
slices=[df.loc[:,'the'].mean(),df.loc[:,'to'].mean(),df.loc[:,'I'].mean(),df.loc[:,'ect'].mean(),df.loc[:,'as'].mean(),df.loc[:,'is'].mean(),df.loc[:,'on'].mean(),df.loc[:,'th'].mean(),df.loc[:,'you'].mean(),df.loc[:,'please'].mean(),df.loc[:,'in'].mean(),df.loc[:,'or'].mean()]
lbl=['the','to','I','ect','as','is','on','th','you','please','in','or']

plt.bar(lbl,slices, color='r',width=0.8)
plt.xlabel('Words')
plt.ylabel('Count')
plt.title('Email is Spam or Not')
plt.show()

#Spliting the DataSet
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#Training Variables
print("x_train -")
print(x_train)
print("y_train -")
print(y_train)

#Testing Variables
print("x_test -")
print(x_test)
print("y_test -")
print(y_test)

#Applying Logistic Regression
from sklearn.linear_model import LogisticRegression

classifier=LogisticRegression(random_state=0)

#Training the Model
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

#Performance messures of the model 
from sklearn.metrics import confusion_matrix, accuracy_score

#Confusion Matrix
cm=confusion_matrix(y_test,y_pred)

#Accuracy Score
ac=accuracy_score(y_test,y_pred)*100

print("Confusion Matrix - \n")
print(cm)

print("Accuracy Score - ")
print(ac)


