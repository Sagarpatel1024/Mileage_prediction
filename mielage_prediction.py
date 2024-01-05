import pandas as pd

mielage = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/MPG.csv')

mielage.head()

mielage.info()

mielage.describe()

mielage.columns

y = mielage['mpg']
x = mielage[['cylinders', 'displacement', 'weight',
       'acceleration','model_year']]

from sklearn.model_selection import train_test_split

x_train ,x_test ,y_train ,y_test = train_test_split(x,y,random_state=2529)

x_train.shape ,x_test.shape ,y_train.shape ,y_test.shape

from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(x_train,y_train)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_absolute_percentage_error

mean_absolute_percentage_error(y_test , y_predict)
