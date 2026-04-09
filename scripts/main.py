import src.C1.data_loader as dl
from src.C4.train_ml_models import run_ml
from src.C3.hyperparameter_search import hyperparameter_search
from sklearn.metrics import accuracy_score, f1_score
import os
import pickle

cwd = os.getcwd()

### ============= 1. DATA LOADING ============= ###
print("Cargando datos...")
feat_mat, stats_report = dl.dataloader(cwd=cwd)

### ============= 2. MACHINE LEARNING ============= ###
print("\nEntrenando modelos base...")
# Recibimos también los datos de TEST que Dana acaba de habilitar
dict_ml, X_train, y_train, X_test, y_test = run_ml(feat_mat, cwd)

acc_base = dict_ml['accuracy']
print(f"Resultado Modelo Base: {acc_base:.4f}")

### ============= 3. MODEL OPTIMIZATION ============= ###
print("\nIniciando búsqueda de mejores hiperparámetros...")
limites = dict_ml['bounds_recomendados'] 

# Tu módulo C3 hace su magia
dict_opt = hyperparameter_search(X_train, y_train, dict_ml, limites)
modelo_final = dict_opt['modelo_optimizado']

### ============= 4. EVALUACIÓN FINAL (La prueba de fuego) ============= ###
print("\n=== VALIDACIÓN FINAL EN DATOS NO VISTOS ===")

# Probamos el modelo que TÚ optimizaste contra el X_test de Dana
y_pred = modelo_final.predict(X_test)
acc_final = accuracy_score(y_test, y_pred)
f1_final = f1_score(y_test, y_pred, average='macro')

print(f"Accuracy final optimizado: {acc_final:.4f}")
print(f"Mejora neta: {((acc_final - acc_base) * 100):+.2f}%")

# Guardamos los resultados finales para Daisy
dict_final_viz = {
    'accuracy_mejorado': acc_final,
    'f1_mejorado': f1_final,
    'incremento': acc_final - acc_base
}

output_dir = os.path.join(cwd, "src", "C3", "outputs")
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "final_optimization_metrics.pkl"), "wb") as f:
    pickle.dump(dict_final_viz, f)

### ============= 5. VISUALIZATION ============= ###
# Aquí es donde Daisy (C6) tomaría el control
print("\nProceso terminado. Resultados listos para reporte.")

