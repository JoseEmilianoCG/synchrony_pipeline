import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

# CAMBIO 1: Importamos differential_evolution en lugar de minimize
from scipy.optimize import differential_evolution

def hyperparameter_search(X_train: pd.DataFrame, y_train: pd.DataFrame, dict_ml: dict, bounds: tuple) -> dict:
    assert len(X_train) == len(y_train), "ERROR: Discrepancia en el número de muestras"
    
    modelo_base = dict_ml['mejor_modelo']
    nombres_params = dict_ml['hyperparams_names'] 
    
    # TRUCO PRO: Variables para rastrear el récord (el mejor error)
    mejor_loss_actual = float('inf')
    history = []

    def f(params):
        nonlocal mejor_loss_actual # Permite modificar la variable de afuera
        
        # TRADUCTOR CONTINUO -> DISCRETO
        config = {}
        for i, name in enumerate(nombres_params):
            if name in ['n_estimators', 'max_depth']:
                config[name] = int(round(params[i])) 
            else:
                config[name] = float(params[i]) 
        
        modelo_iter = clone(modelo_base)
        modelo_iter.set_params(**config)
        
        # Hacemos el cross validation
        scores = cross_val_score(modelo_iter, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        loss = -scores.mean()
        
        # TRUCO PRO: Solo actualizamos la curva si encontramos un error MÁS BAJO
        if loss < mejor_loss_actual:
            mejor_loss_actual = loss
            
        history.append(mejor_loss_actual) # La gráfica ahora será una escalera perfecta hacia abajo
        return loss

    # CAMBIO 2: Usamos el algoritmo genético (differential_evolution)
    # popsize y maxiter controlan qué tanto busca. Mantenemos números bajos para que tu PC no explote.
    res = differential_evolution(f, bounds=bounds, maxiter=5, popsize=3, seed=42)

    if not res.success:
        print("El optimizador no convergió totalmente.")

    # TRADUCTOR PARA EL RESULTADO FINAL
    best_config = {}
    for i, name in enumerate(nombres_params):
        if name in ['n_estimators', 'max_depth']:
            best_config[name] = int(round(res.x[i]))
        else:
            best_config[name] = float(res.x[i])

    print(f"Mejores hiperparámetros: {best_config}")

    modelo_optimizado = clone(modelo_base).set_params(**best_config)
    modelo_optimizado.fit(X_train, y_train)

    # =================================================================
    # Generar y guardar la gráfica de la Curva de Convergencia
    # =================================================================
    output_dir = os.path.join("src", "C3", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 5))
    # Usamos drawstyle='steps-post' para que se vea como una escalera de progreso real
    plt.plot(history, color='green', linewidth=2.5, drawstyle='steps-post')
    plt.title('Curva de Convergencia', fontsize=14)
    plt.xlabel('Iteraciones', fontsize=12)
    plt.ylabel('Error %', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    img_path = os.path.join(output_dir, "curva_mejora.png")
    plt.savefig(img_path)
    plt.close() 
    print(f"Gráfica guardada en: {img_path}\n")
    # =================================================================

    dict_opt = {
        'modelo_optimizado': modelo_optimizado,
        'history': history,
        'best_params': best_config
    }
    return dict_opt