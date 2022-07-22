import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

startup_df=pd.read_csv('trial.csv')
startup_df.drop(startup_df.columns[[0,3]], axis=1, inplace=True)
startup_df['total_supply'].fillna(value=startup_df['total_supply'].mean(), inplace=True)
startup_df['total_sales'].fillna(value=startup_df['total_sales'].mean(), inplace=True)
startup_df['volume'].fillna(value=startup_df['volume'].mean(), inplace=True)
startup_df['max_price'].fillna(value=startup_df['max_price'].mean(), inplace=True)
startup_df['floor_price'].fillna(value=startup_df['floor_price'].mean(), inplace=True)
startup_df['holders'].fillna(value=startup_df['holders'].mean(), inplace=True)
startup_df['buyers'].fillna(value=startup_df['buyers'].mean(), inplace=True)
startup_df['sellers'].fillna(value=startup_df['sellers'].mean(), inplace=True)
startup_df['liquidity'].fillna(value=startup_df['liquidity'].mean(), inplace=True)
startup_df['marketcap'].fillna(value=startup_df['marketcap'].mean(), inplace=True)
startup_df['total_transfers'].fillna(value=startup_df['total_transfers'].mean(), inplace=True)
startup_df['avg_community_sentiment'].fillna(value=startup_df['avg_community_sentiment'].mean(), inplace=True)


startup_df['total_supply'] = startup_df['total_supply'].astype('Float32')
startup_df['total_sales'] = startup_df['total_sales'].astype('Float32')
startup_df['volume'] = startup_df['volume'].astype('Float32')
startup_df['max_price'] = startup_df['max_price'].astype('Float32')
startup_df['floor_price'] = startup_df['floor_price'].astype('Float32')
startup_df['holders'] = startup_df['holders'].astype('Float32')
startup_df['buyers'] = startup_df['buyers'].astype('Float32')
startup_df['sellers'] = startup_df['sellers'].astype('Float32')
startup_df['liquidity'] = startup_df['liquidity'].astype('Float32')
startup_df['marketcap'] = startup_df['marketcap'].astype('Float32')
startup_df['total_transfers'] = startup_df['total_transfers'].astype('Float32')
startup_df['avg_community_sentiment'] = startup_df['avg_community_sentiment'].astype('Float32')

shape=startup_df.shape
print("Dataset contains {} rows and {} columns".format(shape[0],shape[1]))

x=startup_df.iloc[:,:10]
y=startup_df.iloc[:,11]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

print("X_train:",x_train.shape)
print("X_test:",x_test.shape)
print("Y_train:",y_train.shape)
print("Y_test:",y_test.shape)

linreg=LinearRegression()
linreg.fit(x_train,y_train)

y_pred=linreg.predict(x_test)

Accuracy=r2_score(y_test,y_pred)*100
print(" Accuracy of the model is %.2f" %Accuracy)

pred_df=pd.DataFrame({'Actual Value':y_test,'Predicted Value':y_pred,'Difference':y_test-y_pred})

#plt.scatter(pred_df.index.values, pred_df['Actual Value'], c='b', marker='x', label='1')
#plt.scatter(pred_df.index.values, pred_df['Predicted Value'], c='r', marker='s', label='-1')
import plotly.express as px
#df = px.data.iris()
#fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species",
                # size='petal_length', hover_data=['petal_width'])
#fig.show()
import plotly.graph_objects as go
fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=pred_df.index.values, y=pred_df['Actual Value'], mode='markers', name='Actual Values'))
fig.add_trace(go.Scatter(x=pred_df.index.values, y=pred_df['Predicted Value'], mode='markers', name='Predicted Valued'))

#fig.show()
figjson=fig.to_json()
#pred_df.to_json('LinearRegression.json')