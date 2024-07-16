# %%
import pandas as pd
import plotly.express as px
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from IPython.display import display
# %%
train_df = pd.read_csv(r"study_ai_ml_google\data\titanic_train.csv")
test_df = pd.read_csv(r"study_ai_ml_google\data\titanic_test.csv")
display(train_df.head(5))
display(test_df.head(5))
# %%


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    replace_dict = {"male": 0, "female": 1}
    df = data[["Sex", "Age"]].copy()
    df["Sex"] = df["Sex"].apply(lambda x: replace_dict[x])
    df["Age"] = df["Age"].fillna(df["Age"].median(numeric_only=True))

    return df


display(train_df.describe())
display(train_df.isnull().sum())
label_data = train_df[["Survived"]].copy()
train_data = preprocess(train_df)
display(train_data.isnull().sum())
print(train_data["Age"].mean(), train_data["Age"].var())
# %%
X_train, X_test, y_train, y_test = train_test_split(
    train_data, label_data, test_size=0.2, random_state=37
)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression())
])
pipeline.fit(X_train, y_train.values.ravel())
y_pred = pipeline.predict(X_test)
# %%
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
logreg = pipeline.named_steps["logreg"]
coefficients = logreg.coef_
intercept = logreg.intercept_
print("Coefficients:", coefficients)
print("Intercept:", intercept)
# %%
scaler = pipeline.named_steps['scaler']
print(scaler.__dict__)
# %%
data = pd.DataFrame({"Sex": [0], "Age": [30], })
print(pipeline.predict(data))
print(pipeline.predict_proba(data))
# %%
