import src.C1.data_loader as dl
from src.C4.train_ml_models import run_ml
from src.C3.hyperparameter_search import hyperparameter_search
import os



cwd = os.getcwd()

### =============DATA LOADING=============== ###
feat_mat, stats_report = dl.dataloader(cwd=cwd)

### =============MACHINE LEARNING============== ###
run_ml(feat_mat,cwd)
### =============MODEL OPTIMIZATION=============== ###
##### dict_ml, X_train, y_train = run_ml(feat_mat, cwd)
#### limites = dict_ml['bounds_recomendados'] los limites donde estan?
dict_opt = hyperparameter_search(X_train, y_train, dict_ml, limites)
### =============VISUALIZATION=============== ###

