from sklearn.metrics import accuracy_score, f1_score
import pickle
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from scipy.optimize import differential_evolution


def hyperparameter_search(X_train, y_train, dict_ml, bounds):

    assert len(X_train) == len(y_train), "ERROR: Discrepancia en muestras"

    modelo_base = dict_ml['mejor_modelo']
    nombres_params = dict_ml['hyperparams_names']

    mejor_loss_actual = float('inf')
    history = []
    patience = 10
    no_improve = 0

    def f(params):
        nonlocal mejor_loss_actual, no_improve

        config = {}
        for i, name in enumerate(nombres_params):
            if name in ['n_estimators', 'max_depth']:
                config[name] = int(round(params[i]))
            else:
                config[name] = float(params[i])

        modelo_iter = clone(modelo_base)
        modelo_iter.set_params(**config)

        # 🔥 MÉTRICA CORRECTA
        scores = cross_val_score(
            modelo_iter,
            X_train,
            y_train,
            cv=5,
            scoring='f1_macro',
            n_jobs=-1
        )

        f1 = scores.mean()
        loss = 1 - f1  # minimizamos

        if loss < mejor_loss_actual:
            mejor_loss_actual = loss
            no_improve = 0
        else:
            no_improve += 1

        history.append(1 - mejor_loss_actual)  # guardamos F1

        # 🔥 early stopping manual (suave)
        if no_improve >= patience:
            return mejor_loss_actual

        return loss

    res = differential_evolution(
        f,
        bounds=bounds,
        maxiter=20,
        popsize=10,
        seed=42
    )

    # =============================
    # Mejor configuración
    # =============================
    best_config = {}
    for i, name in enumerate(nombres_params):
        if name in ['n_estimators', 'max_depth']:
            best_config[name] = int(round(res.x[i]))
        else:
            best_config[name] = float(res.x[i])

    print(f"Mejores hiperparámetros: {best_config}")

    modelo_optimizado = clone(modelo_base).set_params(**best_config)
    modelo_optimizado.fit(X_train, y_train)

    # =============================
    # Curva de convergencia
    # =============================
    output_dir = os.path.join("src", "C3", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history, linewidth=2.5, drawstyle='steps-post')
    plt.title('Curva de Convergencia (F1 Macro)', fontsize=14)
    plt.xlabel('Iteraciones', fontsize=12)
    plt.ylabel('F1 Macro', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.savefig(os.path.join(output_dir, "curva_convergencia.png"))
    plt.close()

    return {
        'modelo_optimizado': modelo_optimizado,
        'history': history,
        'best_params': best_config
    }


# =====================================
# Evaluación final
# =====================================
def evaluar_modelo_final(modelo_final, X_test, y_test, acc_base, cwd):

    print("\n=== VALIDACIÓN FINAL EN DATOS NO VISTOS ===")

    y_pred = modelo_final.predict(X_test)

    acc_final = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    print(f"Accuracy final: {acc_final:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
    print(f"Mejora accuracy: {((acc_final - acc_base)*100):+.2f}%")

    results = {
        'accuracy': acc_final,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'incremento': acc_final - acc_base
    }

    output_dir = os.path.join(cwd, "src", "C3", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "final_metrics.pkl"), "wb") as f:
        pickle.dump(results, f)

    return results