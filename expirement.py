"""
classification_task.py
Train either a Decision Tree or Random Forest on the spam dataset.
Splits the first 1000 rows for training and the remaining 3601 for testing.
Handles missing values using column-wise means.
Allows dynamic selection of key hyperparameters.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Function to run expirements with different parameters.
def run_classification(
    csv_path: str,
    label_col: str,
    model_type: str = "tree",
    criterion: str = "gini",
    n_estimators: int = 500,
    max_features: str = "sqrt",
    train_size: int = 1000,
    random_state: int = 42,
):
    """
    Params for the run_classification function
    ----------
    csv_path : str
        Path to the dataset CSV.
    label_col : str
        Name of the label column (e.g., 'class' or 'label').
    model_type : str
        'tree' for Decision Tree or 'forest' for Random Forest.
    criterion : str
        Splitting criterion ('gini' or 'entropy').
    n_estimators : int
        Number of trees for Random Forest (ignored for Decision Tree).
    max_features : str
        Number of features to consider at each split ('auto', 'sqrt', 'log2').
    train_size : int
        Number of instances used for training (rest used for testing).
    random_state : int
        Random seed for reproducibility.
    """

    # Load dataset 
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {csv_path}, shape={df.shape}")

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset columns.")

    if train_size >= len(df):
        raise ValueError(f"train_size={train_size} >= dataset length {len(df)}")

    # Split dataset 
    df_train = df.iloc[:train_size].copy()
    df_test = df.iloc[train_size:].copy()

    X_train = df_train.drop(columns=[label_col])
    y_train = df_train[label_col]
    X_test = df_test.drop(columns=[label_col])
    y_test = df_test[label_col]

    # Handle missing values (column-wise means)
    imputer = SimpleImputer(strategy="mean")

    # Model selection (based on whatever the user inputs into the run_classification function)
    if model_type.lower() == "tree":
        clf = DecisionTreeClassifier(criterion=criterion, random_state=random_state)
        model_name = f"Decision Tree ({criterion})"

    elif model_type.lower() == "forest":
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,
        )
        model_name = f"Random Forest ({criterion}, n={n_estimators}, maxf={max_features})"

    else:
        raise ValueError("model_type must be 'tree' or 'forest'")

    pipe = Pipeline([
        ("imputer", imputer),
        ("clf", clf)
    ])

    # Train
    print(f"\nTraining {model_name} with {train_size} training samples ...")
    pipe.fit(X_train, y_train)

    # Predict
    y_pred = pipe.predict(X_test)

    # Stats 
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    print(f"\n=== {model_name} Results ===")
    print(f"Overall Accuracy: {acc:.4f}")
    print("Per-class Accuracy:")
    for lbl, score in zip(np.unique(y_test), per_class_acc):
        print(f"  {lbl}: {score:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report)


    # Return metrics
    return {
        "model": model_name,
        "accuracy": acc,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": cm,
        "report": report,
    }


# Main function to run tests
if __name__ == "__main__":
    # Change parameters here to test different configurations
    run_classification(
        csv_path="spam.csv",
        label_col="Class",
        model_type="forest",  
        criterion="gini",  
        n_estimators=5000,     
        max_features="sqrt",   
        train_size=1500,      
        random_state=42,
    )
