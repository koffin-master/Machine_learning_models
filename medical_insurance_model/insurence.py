import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer

insurenceDf = pd.read_csv("/Users/rahmani/Documents/Assets_ML/insurance.csv")


num_cols = insurenceDf.select_dtypes(include=["number"])
num_cols = num_cols.drop("charges",axis=1)
print(num_cols.columns)
cat_cols = ["sex", "region","smoker"]

# print(insurenceDf.columns)

input_col = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

X = insurenceDf[input_col]
y = insurenceDf["charges"]

insure_train , insure_test, y_train, y_test = train_test_split(X,y,train_size=0.7,random_state=42)

pipelines = {}

models = {
    "XGB" : XGBRegressor(n_estimators=200,random_state=42),
    "Random_forest" : RandomForestRegressor(n_estimators=200,random_state=42),
    "Decision_Tree" : DecisionTreeRegressor(random_state=42)
}

preprocessor = ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(),num_cols.columns),
        ("cat", OneHotEncoder(),cat_cols)
    ]
)

for name , model in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    pipe.fit(insure_train,y_train)
    pipelines[name] = pipe

for name, pipe in pipelines.items():
    y_pred = pipe.predict(insure_test)
    print(f"Root mean square error for {name}: ",root_mean_squared_error(y_test,y_pred))
    print(f"R2 score : {name}", r2_score(y_test,y_pred))
    print()