import src.C1.data_loader as dl
from src.C4.train_ml_models import run_ml
import os



cwd = os.getcwd()

### =============DATA LOADING=============== ###
feat_mat, stats_report = dl.dataloader(cwd=cwd)

### =============MACHINE LEARNING============== ###
run_ml(feat_mat,cwd)
### =============MODEL OPTIMIZATION=============== ###

### =============VISUALIZATION=============== ###

