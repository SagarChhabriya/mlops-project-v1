import os
import pandas as pd
import joblib
import wandb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from requests.exceptions import HTTPError

import warnings
warnings.filterwarnings('ignore')


wandb.login(key=os.getenv("WANDB_API_KEY"))
wandb.init(project="mlops-project-v2", job_type="training")
print("✅ W&B run started:", wandb.run.name)

# --- Global Configuration and Data Loading ---

PROJECT = "mlops-project-v2"
os.environ["WANDB_AGENT_DISABLE_FLAPPING"] = "true"

# Define a fallback configuration globally (simple alternative)
FALLBACK_CONFIG = {
    "C": 1.0, 
    "penalty": "elasticnet", 
    "solver": "saga",
    "l1_ratio": 0.5, 
    "max_iter": 100
}


def log_dataset(filepath="customer_churn.csv"):
    """
    Loads, cleans the data, and logs it as a W&B Artifact.
    Returns the cleaned feature matrix (X) and target vector (y).
    This is just a dummy text to be deleted. 
    """
    print(f"\n1. Logging Dataset as W&B Artifact from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: '{filepath}' not found. Please ensure the file is in the script's directory.")
        exit()

    # Fix feature names - remove extra spaces and standardize
    df.columns = df.columns.str.strip().str.replace(r'[\s\xa0]+', ' ', regex=True)
    
    # Split features/target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {X.columns.tolist()}")

    # Log the dataset artifact
    run_data_versioning = wandb.init(project=PROJECT, job_type="data-versioning")
    dataset_artifact = wandb.Artifact(
        name="customer-churn-dataset",
        type="dataset",
        description="Telecom churn dataset for classification project",
        metadata={
            "rows": df.shape[0],
            "features": list(X.columns),
            "target": "Churn",
            "source": filepath
        }
    )
    dataset_artifact.add_file(filepath)
    wandb.log_artifact(dataset_artifact)
    run_data_versioning.finish()
    print("Dataset artifact logged.")
    
    return X, y


def prepare_data(X, y):
    """
    Splits the data into training/testing sets and applies standard scaling.
    Returns scaled data splits and the fitted scaler.
    """
    print("\n2. Preparing data (Splitting and Scaling)...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler and feature names for later use in prediction
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(list(X.columns), "models/feature_names.pkl")
    
    return X_train_scaled, X_test_scaled, y_train, y_test


def sweep_train():
    """
    Trains a Logistic Regression model using parameters provided by the W&B sweep,
    evaluates it, and logs the metrics to W&B.
    This function is executed by the wandb.agent.
    """
    # NOTE: Relies on global X and y for simplicity in single-script conversion.
    global X, y
    
    run = wandb.init(project=PROJECT)
    config = wandb.config

    # Split and scale data (replicated inside sweep_train for isolation, using globals)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Instantiate model with sweep parameters
    model = LogisticRegression(
        C=config.C,
        penalty=config.penalty,
        solver=config.solver,
        max_iter=config.max_iter,
        l1_ratio=config.l1_ratio if config.penalty == "elasticnet" else None,
        random_state=42
    )

    # Train and evaluate
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    # Log metrics
    wandb.log({
        "val_accuracy": accuracy_score(y_test, y_pred),
        "val_roc_auc": roc_auc_score(y_test, y_prob), 
        "val_precision": precision_score(y_test, y_pred),
        "val_recall": recall_score(y_test, y_pred)
    })
    
    # Store parameters locally in a CSV file
    params_data = {
        'run_id': run.id,
        'C': config.C,
        'penalty': config.penalty,
        'solver': config.solver,
        'max_iter': config.max_iter,
        'l1_ratio': config.l1_ratio,
        'auc': auc
    }

    # Append to CSV file
    df = pd.DataFrame([params_data])
    if not os.path.exists('sweep_results.csv'):
        df.to_csv('sweep_results.csv', index=False)
    else:
        df.to_csv('sweep_results.csv', mode='a', header=False, index=False)

    run.finish()


def run_sweep(sweep_config):
    """
    Defines the W&B sweep, runs the agent for optimization, and returns the sweep ID.
    """
    print("\n3. Defining and Running W&B Sweep (5 trials)...")
    
    
    sweep_id = wandb.sweep(sweep_config, project=PROJECT)
    print(f"Created Sweep ID: {sweep_id}")

    # Run the agent, which executes the sweep_train function multiple times
    wandb.agent(sweep_id, function=sweep_train, count=5)
    print("Sweep complete.")
    
    return sweep_id


def get_best_run_config(sweep_id):
    """
    Read best parameters from local CSV file
    """
    # print("\n4. Fetching best parameters from W&B (via Cloud API)...")
    
    # api = wandb.Api()
    
    # Default parameters in case anything fails
    fallback_config = {
        "C": 1.0, "penalty": "elasticnet", "solver": "saga",
        "l1_ratio": 0.5, "max_iter": 100
    }
    
    
    print("\n4. Reading best parameters from local CSV...")
    
    try:
        # Read the CSV with all sweep results
        df = pd.read_csv('sweep_results.csv')
        
        # Find the row with highest AUC
        best_row = df.loc[df['auc'].idxmax()]
        
        best_config = {
            "C": best_row['C'],
            "penalty": best_row['penalty'],
            "solver": best_row['solver'], 
            "max_iter": best_row['max_iter'],
            "l1_ratio": best_row['l1_ratio']
        }
        
        print(f"✅ Best run AUC = {best_row['auc']:.4f}")
        print("Best parameters:", best_config)
        return best_config
        
    except Exception as e:
        print(f"⚠️ Could not read local CSV: {e}")
        print("Using default config instead.")
        return fallback_config


def train_and_log_best_model(X_train, X_test, y_train, y_test, best_config, sweep_id):
    """
    Retrains the final model using the best hyperparameters (from best_config)
    and logs the final model artifact.
    """
    # Use .get() on the best_config to safely retrieve parameters, defaulting to FALLBACK_CONFIG values
    print("\n5. Retraining and Logging Final Best Model...")
    
    BEST_C = best_config.get("C", FALLBACK_CONFIG["C"])
    BEST_PENALTY = best_config.get("penalty", FALLBACK_CONFIG["penalty"])
    BEST_SOLVER = best_config.get("solver", FALLBACK_CONFIG["solver"])
    BEST_L1_RATIO = best_config.get("l1_ratio", FALLBACK_CONFIG["l1_ratio"])
    BEST_MAX_ITER = best_config.get("max_iter", FALLBACK_CONFIG["max_iter"])

    # Start the final W&B run
    run_tuned = wandb.init(project=PROJECT, job_type="model-tuning", config={
        "C": BEST_C, 
        "penalty": BEST_PENALTY, 
        "solver": BEST_SOLVER, 
        "l1_ratio": BEST_L1_RATIO, 
        "max_iter": BEST_MAX_ITER,
        "source_sweep_id": sweep_id
    })

    # Train the final tuned model using the provided configuration
    best_model = LogisticRegression(
        C=BEST_C,
        penalty=BEST_PENALTY,
        solver=BEST_SOLVER,
        l1_ratio=BEST_L1_RATIO,
        max_iter=BEST_MAX_ITER,
        random_state=42
    )
    best_model.fit(X_train, y_train)

    # Final evaluation of the tuned model
    y_pred_tuned = best_model.predict(X_test)
    y_prob_tuned = best_model.predict_proba(X_test)[:, 1]
    
    tuned_metrics = {
        "final_accuracy": accuracy_score(y_test, y_pred_tuned),
        "final_roc_auc": roc_auc_score(y_test, y_prob_tuned),
        "final_precision": precision_score(y_test, y_pred_tuned),
        "final_recall": recall_score(y_test, y_pred_tuned)
    }
    wandb.log(tuned_metrics)
    print("Tuned Model Performance:")
    for k, v in tuned_metrics.items():
        print(f"{k}: {v:.4f}")

    from datetime import datetime
    # Create a timestamped filename for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"models/logistic_regression_tuned_{timestamp}.pkl"

    # Save the model
    joblib.dump(best_model, model_filename)
    print(f"Model saved to {model_filename}")


    tuned_artifact = wandb.Artifact(
        name="logistic-regression-tuned",
        type="model",
        description="Tuned Logistic Regression via W&B Sweep",
        metadata={
            "source_sweep": sweep_id, 
            "hyperparameters": run_tuned.config.as_dict()
        }
    )
    tuned_artifact.add_dir("models")
    wandb.log_artifact(tuned_artifact)
    
    # Link artifact to a common alias for easy retrieval
    wandb.run.link_artifact(tuned_artifact, f"{PROJECT}/logistic-regression-model:latest")

    run_tuned.finish()
    print("Tuned model artifact logged and linked.")


def main():
    """
    Main execution pipeline for the churn prediction training process.
    """
    
    # Define Sweep Configuration
    sweep_config = {
        "method": "random",
        "metric": {"name": "val_roc_auc", "goal": "maximize"},
        "parameters": {
            "C": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e4},
            "penalty": {"values": ["elasticnet"]},
            "solver": {"values": ["saga"]},
            "max_iter": {"values": [100, 200, 500]},
            "l1_ratio": {"distribution": "uniform", "min": 0.0, "max": 1.0}
        }
    }
    
    # 1. Log Dataset (X and y are set globally here for sweep_train()'s access)
    global X, y
    X, y = log_dataset()
    
    # 2. Prepare Data
    X_train_scaled, X_test_scaled, y_train, y_test = prepare_data(X, y)
       
    # 3. Run Sweep
    sweep_id = run_sweep(sweep_config)
    
    # 4. Fetch Best Parameters
    best_config = get_best_run_config(sweep_id)

    # 5. Retrain and Log Best Model
    train_and_log_best_model(
        X_train_scaled, 
        X_test_scaled, 
        y_train, 
        y_test, 
        best_config, 
        sweep_id
    )


if __name__ == "__main__":
    # Ensure you are logged into W&B before running (e.g., `wandb login`)
    main()
