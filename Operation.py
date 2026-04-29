# equipment_utilization.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
df = pd.read_csv("data/equipment.csv")
X = df[["scheduled_hours", "downtime_hours", "load_lbf"]]
y = df["utilization"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Equipment Utilization R²:", r2_score(y_test, y_pred))
#Prediction
model.predict([[8, 1.5, 250]])

# budget_variance.py
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
df = pd.read_csv("data/budget.csv")
X = df[["production_units", "labor_cost", "material_cost"]]
y = df["budget_variance"]
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = XGBRegressor(n_estimators=200, max_depth=4)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Budget MAE:", mean_absolute_error(y_test, y_pred))
#Prediction
model.predict([[1400, 390000, 450000]])

# workforce_planning.py
import pandas as pd
from sklearn.linear_model import LinearRegression
df = pd.read_csv("data/workforce.csv")
X = df[["output_units", "absenteeism_rate"]]
y = df["workers_needed"]
model = LinearRegression()
model.fit(X, y)
print("Workforce planning model trained")
#Prediction
model.predict([[220, 0.05]])

# resource_allocation.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv("data/resources.csv")
X = df[["labor_hours", "material_kg"]]
y = df["product_units"]
model = RandomForestRegressor(n_estimators=200)
model.fit(X, y)
predicted_units = model.predict([[280, 460]])
print("Predicted production:", predicted_units)
from scipy.optimize import linprog
cost = [50, 30] # labor, material
A = [[-1, -1]]
b = [-100] # demand
bounds = [(0,500),(0,600)]
result = linprog(cost, A_ub=A, b_ub=b, bounds=bounds)
print(result.x)

from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(df["Total_Sales"], order=(1,1,1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=6)
print(forecast)

#Requirements.txt
#pandas
#numpy
#scikit-learn
#xgboost
#scipy




