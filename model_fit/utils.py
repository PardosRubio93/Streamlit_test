import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("../data/train.csv", index_col=0)

y = df['Survived']
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]
X_train, X_test, y_train, y_test = train_test_split(X,y)

# age_means = df.groupby(['Pclass','Sex'])['Age'].mean()
age_mean = X_train['Age'].mean()

