import pickle
import os
from sklearn.metrics import classification_report, accuracy_score, precision_score

def run_ml(feat_mat, cwd):

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC

    # 1- Data
    X = feat_mat.drop("Target", axis=1)
    y = feat_mat["Target"]

    # 2- Asserts
    assert len(X) == len(y), f"ERROR : X has {len(X)} samples but y has {len(y)}"

    # 3- Split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 4- Create the models
    models = {
        "RandomForest":     RandomForestClassifier(),
        "SVM":              SVC(),
        "GradientBoosting": GradientBoostingClassifier(),
    }

    # 5- MODELS
    resultados_modelos = {}

    for model_name, model in models.items():

        # 5.1 Train the models
        model.fit(X_train, y_train)

        # 5.2 Predictions
        y_pred = model.predict(X_test)

        # 5.3 Calculate metrics of the models
        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        report    = classification_report(y_test, y_pred, output_dict=True)

        # 5.4 Save the metrics
        resultados_modelos[model_name] = {
            "model":     model,
            "accuracy":  accuracy,
            "precision": precision,
            "report":    report,
        }

    # 6- Compare all the models
    print("\n=== COMPARE THE MODELS ===")
    print(f"{'Modelo':<20} {'Accuracy':>10} {'Precision':>10}")
    print("-" * 42)
    for name, res in resultados_modelos.items():
        print(f"{name:<20} {res['accuracy']:>10.4f} {res['precision']:>10.4f}")

    # 7- Identify the best models
    best_name = max(resultados_modelos, key=lambda m: resultados_modelos[m]["accuracy"])
    best_model = resultados_modelos[best_name]["model"]
    print(f"\n Best model: {best_name} (Accuracy: {resultados_modelos[best_name]['accuracy']:.4f})")

    # 8- Save the outputs
    output_dir = os.path.join(cwd, "src", "C4", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Save all the metrics for all the models
    all_metrics = {k: {"accuracy": v["accuracy"], "precision": v["precision"], "report": v["report"]}
                   for k, v in resultados_modelos.items()}

    with open(os.path.join(output_dir, "all_metrics.pkl"), "wb") as f:
        pickle.dump(all_metrics, f)

   # Save all the models
    for name, res in resultados_modelos.items():
        model_path = os.path.join(output_dir, f"{name}_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(res["model"], f)
        print(f" {name}_model.pkl guardado")

    # Save the best model
    best_model_obj = resultados_modelos[best_name]["model"]
    output_path = os.path.join(output_dir, "best_model.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(best_model_obj, f)

    if best_name == "RandomForest":
        nombres_params = ['n_estimators', 'max_depth']
        bounds = ((10, 200), (2, 20))
    elif best_name == "SVM":
        nombres_params = ['C', 'gamma']
        bounds = ((0.1, 100.0), (0.001, 1.0))
    elif best_name == "GradientBoosting":
        nombres_params = ['n_estimators', 'learning_rate']
        bounds = ((50, 300), (0.01, 0.5))

    f1_ganador = resultados_modelos[best_name]['report']['macro avg']['f1-score']

    dict_ml = {
        'mejor_modelo': best_model_obj,
        'hyperparams_names': nombres_params,
        'bounds_recomendados': bounds,
        'f1_score': f1_ganador,
        'accuracy': resultados_modelos[best_name]['accuracy']
    }

    return dict_ml, X_train, y_train, X_test, y_test 





