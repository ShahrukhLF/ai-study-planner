import pandas as pd
import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import os

# Create directories if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

def load_data():
    study_data = pd.read_csv("data/study_data.csv")
    user_history = pd.read_csv("data/user_history.csv")
    return study_data, user_history

def train_models():
    study_data, user_history = load_data()

    # Feature Engineering
    X_study = study_data[["difficulty", "importance", "time_required"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_study)

    # KMeans Clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    # Cluster Order by Difficulty
    cluster_order = np.argsort(kmeans.cluster_centers_[:, 0])

    # KNN Regression
    X_user = user_history[["cluster", "study_time"]]
    y_user = user_history["days_needed"]
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_user, y_user)

    # Linear Regression
    X_train, X_test, y_train, y_test = train_test_split(
        X_study, study_data["time_required"], test_size=0.2, random_state=42
    )
    linear_reg = LinearRegression()
    linear_reg.fit(X_train, y_train)

    # Evaluation
    y_pred = linear_reg.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Save Models
    joblib.dump(kmeans, "models/kmeans_model.pkl")
    joblib.dump(knn, "models/knn_model.pkl")
    joblib.dump(linear_reg, "models/linear_reg_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(cluster_order, "models/cluster_order.pkl")

    # Save Accuracy
    with open("models/accuracy.txt", "w") as f:
        f.write(f"R² Score: {r2:.2f}, RMSE: {rmse:.2f}")

    return kmeans, knn, linear_reg, X_test, y_test

if __name__ == "__main__":
    train_models()
    print("Models trained and saved successfully!")
