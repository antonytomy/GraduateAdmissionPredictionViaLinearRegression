



import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import tensorflow as tf
import seaborn as sns
from sklearn import metrics
import sklearn
import matplotlib.pyplot as plt
import pickle
get_ipython().run_line_magic('matplotlib', 'inline')



data=pd.read_csv("admission.csv")

data.drop(["Serial No."],axis=1,inplace=True)



data.head()



X=data.drop('Chance of Admit ',axis=1)
y=data["Chance of Admit "]


X_train, X_test, y_train, y_test = train_test_split(X, y.values, test_size=0.3, random_state=101)


model=LinearRegression()
model.fit(X_train,y_train)



predictions = model.predict(X_test)



print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))



pred_df = pd.DataFrame(y_test,columns=['Test Y'])



pred_df.head()




test_predictions = pd.Series(predictions.reshape(150,))



test_predictions.head()



pred_df = pd.concat([pred_df,test_predictions],axis=1)



pred_df.columns = ['Test Y','Model Predictions']




pred_df.head()



sns.scatterplot(x='Test Y',y='Model Predictions',data=pred_df)



filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))





