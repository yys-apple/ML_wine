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