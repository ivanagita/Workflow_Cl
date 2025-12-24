import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    # Load data (asumsi data ada di folder Workflow-CI)
    df = pd.read_csv('dataset_clean.csv')
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=50, max_depth=5)
    
    # Bungkus pakai start_run biar dia tahu ini bagian dari MLProject
    with mlflow.start_run():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        
        # Simpan metric dan folder 'model'
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
        
        print(f"BERHASIL! Akurasi: {acc}")

if __name__ == "__main__":
    main()