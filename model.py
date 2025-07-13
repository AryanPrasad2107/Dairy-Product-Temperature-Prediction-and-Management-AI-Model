import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Step 1: Load the dataset
df = pd.read_csv("advanced_dairy_cold_chain_dataset.csv")

# Step 2: Define features and target
features = [
    'product_type', 'external_temp', 'current_room_temp',
    'humidity', 'volume_kg', 'packaging_type',
    'storage_time_hr', 'airflow_rating'
]
target = 'ideal_room_temp'

X = df[features]
y = df[target]

# Step 3: Preprocess categorical features using OneHotEncoder
categorical_features = ['product_type', 'packaging_type', 'airflow_rating']
numerical_features = [f for f in features if f not in categorical_features]

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(), categorical_features)
], remainder='passthrough')  # Keep numerical features as is

# Step 4: Create a pipeline with preprocessing + model
model_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the model
model_pipeline.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = model_pipeline.predict(X_test)
print("Mean Absolute Error:", round(mean_absolute_error(y_test, y_pred), 2))
print("R² Score:", round(r2_score(y_test, y_pred), 4))

# Step 8: Save the pipeline
joblib.dump(model_pipeline, "advanced_temperature_model.pkl")
print("✅ Model saved as 'advanced_temperature_model.pkl'")
