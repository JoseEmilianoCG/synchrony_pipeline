import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from scipy.optimize import minimize

def hyperparameter_search(X_train: pd.DataFrame, y_train: pd.DataFrame, dict_ml: dict, bounds: tuple) -> dict:
    assert len(X_train) == len(y_train), "ERROR: Discrepancia en el número de muestras"
    
    modelo_base = dict_ml['mejor_modelo']
    nombres_params = dict_ml['hyperparams_names'] 
    
    history = []

    def f(params):
        config = {nombres_params[i]: params[i] for i in range(len(params))}
        
        modelo_iter = clone(modelo_base)
        modelo_iter.set_params(**config)
        
        scores = cross_val_score(modelo_iter, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        loss = -scores.mean()
        
        history.append(loss)
        return loss

    x0 = [np.mean(b) for b in bounds]
    if f(x0) < 0:
        raise ValueError("ERROR: La función objetivo está retornando errores negativos imposibles")

    res = minimize(f, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50})
    
    if not res.success:
        print("AdvertenciaLimiteIteraciones: El optimizador no convergió totalmente.")

    best_config = {nombres_params[i]: res.x[i] for i in range(len(res.x))}
    modelo_optimizado = clone(modelo_base).set_params(**best_config)
    modelo_optimizado.fit(X_train, y_train)

    dict_opt = {
        'modelo_optimizado': modelo_optimizado,
        'history': history,
        'best_params': res.x
    }
    return dict_opt