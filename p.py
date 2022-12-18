
import pandas as pd
import pandas as pd
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from sklearn .model_selection import train_test_split
from sklearn .linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

from sklearn .metrics import classification_report,confusion_matrix,accuracy_score


# Step 1 - Load data and understand
ds = pd.read_csv(r"C:\Users\HP\Downloads\weight-height.csv")
print(ds)
print (ds.head(10))
print (ds.tail(10))
print (ds.describe())
print (ds.isnull().sum())

print('##################################################')
# Step 2 - visualize
x=ds.iloc[:,:-1].values
y=ds.iloc[:,2].values
#plt.bar(x,y,color="green")
#plt.grid(True)
#plt.legend
#plt.show
print('##################################################')
#step 3-convert categoricaldata tinto numerical data

ds['Gender'].replace('Female',0, inplace=True)
ds['Gender'].replace('Male',1, inplace=True)
x = ds.iloc[:, :-1].values
y = ds.iloc[:, 2].values
labelEncoder_gender =  LabelEncoder()
x[:,0] = labelEncoder_gender.fit_transform(x[:,0])

X = np.vstack(x[:, :]).astype(np.float)
print(x)
print(y)
print('##################################################')
#sstep 4-splite

x_train,y_test,x_test,y_train =train_test_split(x,y, test_size=(0.2),random_state=0)

print('##################################################')
#step 5 train
model=LinearRegression()

model.fit ( x_train,y_train)


#step 6 preadic

pre=model.predict(x_test)


#step 7 output
print(classification_report(y_test,pre))
print(accuracy_score(y_test,pre))
print(confusion_matrix(y_test,pre))