import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, root_mean_squared_error,mean_squared_error,mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

resume_df = pd.read_csv("/Users/rahmani/Documents/Assets_ML/resume_job_matching_dataset.csv")

input_col = ["job_description", "resume"]

X = resume_df[input_col]
y = resume_df["match_score"]

resume_train, resume_test, y_train, y_test = train_test_split(X,y,random_state=42,train_size=0.7)

param_grid = {
    "Tree" : {
        "model__max_depth" : [3,5,10],
        "model__min_samples_split" : [2,5,10]
    },
    "Random_Forest" : {
        "model__max_depth" : [3,5,10],
        "model__n_estimators" : [50,100]
    },
    "XGB" : {
        "model__max_depth" : [3,5,10],
        "model__n_estimators" : [50,100]
    }
}

pipelines = {}

models = {
    "Tree" : DecisionTreeRegressor(),
    "Random_Forest" : RandomForestRegressor(),
    "XGB" : XGBRegressor()
}


preprocessor = ColumnTransformer(
    transformers=[
        ("vect", TfidfVectorizer(), "job_description"),
        ("resume", TfidfVectorizer(), "resume")
    ]
)

for name , model in models.items():
    pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
    ])
    grid = GridSearchCV(pipe, param_grid[name],cv=5,scoring="r2")
    grid.fit(resume_train,y_train)
    pipelines[name] = grid

for name, grid in pipelines.items():
    y_pred = grid.predict(resume_test)
    print(f"R2 :{name} \n", r2_score(y_test, y_pred))
    print(f"RMSE :{name} \n", root_mean_squared_error(y_test, y_pred))
    print(f"MSE :{name} \n", mean_squared_error(y_test, y_pred))
    print(f"MAE :{name} \n", mean_absolute_error(y_test, y_pred))

