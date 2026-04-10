import os
from src.C1.data_loader import dataloader
from src.C4.train_ml_models import run_ml
from src.C3.hyperparameter_search import hyperparameter_search, evaluar_modelo_final

from src.visualization.viz import generate_report 

def main():
    cwd = os.getcwd()

    ### ============= 1. DATA LOADING (C1) ============= ###
    print("Cargando datos (C1)...")
    feat_mat, stats_report = dataloader(cwd=cwd)

    ### ============= 2. MACHINE LEARNING (C4) ============= ###
    print("\nEntrenando modelos ml (C4)...")
    dict_ml, X_train, y_train, X_test, y_test = run_ml(feat_mat, cwd)

    ### ============= 3. MODEL OPTIMIZATION (C3) ============= ###
    print("\nIniciando optimización (C3)...")
    dict_opt = hyperparameter_search(X_train, y_train, dict_ml, dict_ml['bounds_recomendados'])

    ### ============= 4. EVALUACIÓN FINAL ============= ###
    evaluar_modelo_final(dict_opt['modelo_optimizado'], X_test, y_test, dict_ml['accuracy'], cwd)

    ### ============= 5. VISUALIZATION Y REPORTE (C6) ============= ###
    print("\nGenerando reporteX (C6)...")
    generate_report() 

if __name__ == "__main__":
    main()

