import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

df = pd.read_csv('merged.csv')
df['male'] = df['Sex'] == 'male'
X = df[['Pclass','male','Age','Fare']]
y = df['Survived']

Median_age_fare = X.fillna({
    'Age':X['Age'].median(),
    'Fare':X['Fare'].median()
})
X = Median_age_fare

X = X[['Pclass','male','Age','Fare']]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=27)

# model = LogisticRegression()
model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Score: ", model.score(X_test, y_test))

y_pred = model.predict(X_test)

print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
print("accuracy: ", accuracy_score(y_test, y_pred)*100,"%")
print("precision: ", precision_score(y_test, y_pred)*100,"%")
print("recall: ", recall_score(y_test, y_pred)*100,"%")
print("F1_score: ", f1_score(y_test, y_pred)*100,"%")