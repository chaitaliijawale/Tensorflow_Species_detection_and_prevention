import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

animal_name = 'Elephant(African)'
df = pd.read_csv('../../data/Population_by_year.csv')
df = df[df['Animal'] == animal_name]
X = df['Year']
Y = df['Population']
#print(X,Y)

Xreshape = X.values.reshape(-1,1)
Yreshape = Y.values.reshape(-1,1)
#X.values.reshape(-1,1)
#Y.values.reshape(-1,1)

X_train, X_test, Y_train, Y_test = train_test_split(Xreshape, Yreshape, test_size=0.15)

model = DecisionTreeClassifier()

model.fit(X_train, Y_train)
#model.fit(X_train.reshape(-1, 1))
#model.fit(Y_train.reshape(-1, 1))
Z = model.predict(X_test)

score = accuracy_score(Y_test, Z)
print(score)
