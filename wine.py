from sklearn.datasets import load_wine
import pandas as pd

# Load dataset
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target
print(df.head())


# Select the two columns of interests
X = df[['alcohol', 'proline']]
print(X.head())

y = df['target']
print(type(y))
print(y.head())

from sklearn.model_selection import train_test_split

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=4
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

model_ss = make_pipeline(
    StandardScaler(),
    SVC()
)

model_ss.fit(X_train, y_train)
acc = model_ss.score(X_test, y_test)

print("Accuracy:", acc)
