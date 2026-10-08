import pandas as pd
df=pd.read_csv('car_price_prediction.csv')
'''
print(df.head(10))
print(df.shape)
print(df.info())
print(df.isnull().sum())'''
print(df["Levy"].value_counts().head(10))
df["Levy"]=df["Levy"].replace("-",pd.NA)
print(df["Levy"].isnull().sum())
df["Levy"]=pd.to_numeric(df["Levy"], errors="coerce")
print(df["Levy"].dtype)
df["Levy"] = df["Levy"].fillna(df["Levy"].median())
print(df["Levy"].isnull().sum())
df["Mileage"]=df["Mileage"].str.replace("km","",regex=False)
df["Mileage"] = pd.to_numeric(df["Mileage"], errors="coerce")
df["Engine volume"]=pd.to_numeric(df["Engine volume"], errors="coerce")
print(df["Engine volume"].dtype)
print(df["Engine volume"].isnull().sum())
df["Engine volume"]=df["Engine volume"].fillna(df["Engine volume"].median())
print(df["Engine volume"].isnull().sum())
print(df["Cylinders"].head(10))
print(df["Cylinders"].dtype)
print(df["Cylinders"].isnull().sum())
categorical_columns = [
    "Manufacturer",
    "Category",
    "Leather interior",
    "Fuel type",
    "Gear box type",
    "Drive wheels",
    "Doors",
    "Wheel",
    "Color"
]
for i in categorical_columns:
    print(i)
    print(df[i].value_counts().head(10))
for j in categorical_columns:
    print(j)
    print(df[j].nunique())
df=df.drop("Model",axis=1)
print("duplicate values:-======================")
print(df.duplicated().sum())
df=df.drop_duplicates()
print(df.duplicated().sum())
print("-----------price check==================")
print(df["Price"].describe())
print((df["Price"] <= 100).sum())
print(df[df["Price"] <= 100][["Price", "Manufacturer", "Prod. year", "Mileage"]].head(20))
df = df[df["Price"] > 100]
print(df["Price"])
print("----------------product year---------------")
print(df["Prod. year"].describe())
print(df["Airbags"].describe())
df=df.drop("ID",axis=1)
print(df.columns)
print("---------checking--------")
print(df[df["Price"] == 26307500])
print(df["Price"].describe())
df=df[df["Price"] != 26307500]
df["Car_age"]=2026-df["Prod. year"]
df=pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True
)
print(df.head())
print("----checking----")
print(df.select_dtypes(include=["object"]).columns)
print(df.isnull().sum().sum())
print(df["Price"].sort_values(ascending=False).head(20))
print(df[df["Price"] > 10000][
    ["Price", "Prod. year", "Mileage"]
])
X=df.drop("Price", axis=1)
y=df["Price"]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
X_train,X_test,y_train,y_test= train_test_split(X,y, test_size=0.2, random_state=42)
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

print("---------------------------------------------")
print(f"{'Model':<20} {'MAE':>12} {'MSE':>16} {'RMSE':>12} {'R2':>8}")
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_prediction = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_prediction)
    mse = mean_squared_error(y_test, y_prediction)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_prediction)
    print(f"{model_name:<20} {mae:>12.2f} {mse:>16.2f} {rmse:>12.2f} {r2:>8.4f}")

model = models["Random Forest"]
print("\n========== ENTER CAR DETAILS ==========")
manufacturer = input("Manufacturer: ")
category = input("Category: ")
leather = input("Leather interior (Yes/No): ")
fuel = input("Fuel type: ")
engine = float(input("Engine volume: "))
mileage = float(input("Mileage: "))
cylinders = float(input("Cylinders: "))
gearbox = input("Gear box type: ")
drive = input("Drive wheels: ")
doors = input("Doors: ")
wheel = input("Wheel: ")
color = input("Color: ")
airbags = float(input("Airbags: "))
levy = float(input("Levy: "))
year = int(input("Production year: "))
print("---------------------------------")
car_age=2026-year
new_car = pd.DataFrame({
    "Levy": [levy],
    "Manufacturer": [manufacturer],
    "Prod. year": [year],
    "Category": [category],
    "Leather interior": [leather],
    "Fuel type": [fuel],
    "Engine volume": [engine],
    "Mileage": [mileage],
    "Cylinders": [cylinders],
    "Gear box type": [gearbox],
    "Drive wheels": [drive],
    "Doors": [doors],
    "Wheel": [wheel],
    "Color": [color],
    "Airbags": [airbags],
    "Car_age": [car_age]
})
new_car=pd.get_dummies(
    new_car,
    columns=categorical_columns,
    drop_first=True
)
new_car=new_car.reindex(columns=X.columns, fill_value=False)
predicted_price = model.predict(new_car)
print("Predicted Price:", predicted_price[0])