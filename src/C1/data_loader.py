from . import preprocessing as prep
from . import processing as proc
from . import stats as st
import json
import os
import pickle


def dataloader(cwd):

    ### =============INPUTS=============== ###
    proc_config_path = os.path.join(cwd, "src", "C1", "inputs", "proc_config.json")
    window_config_path = os.path.join(cwd, "src", "C1", "inputs", "window_config.json")

    # --- Path checks ---
    if not os.path.exists(proc_config_path):
        raise FileNotFoundError("Data Path Error: proc_config.json not found")

    if not os.path.exists(window_config_path):
        raise FileNotFoundError("Data Path Error: window_config.json not found")

    # --- Load configs ---
    with open(proc_config_path, "r") as f:
        proc_config = json.load(f)

    with open(window_config_path, "r") as f:
        window_config = json.load(f)

    # --- Basic config validation ---
    assert len(proc_config["sessions"]) > 0, "Invalid Data: empty sessions"
    assert len(proc_config["stages"]) > 0, "Invalid Data: empty stages"
    assert window_config["winsz"] > 0, "Invalid Data: winsz must be > 0"
    assert window_config["ovlap"] < 1, "Invalid Data: ovlap must be < 1"

    ### =============PREPROCESSING=============== ###
    preprodata = prep.prepro(
        stages=proc_config["stages"], sessions=proc_config["sessions"], cwd=cwd
    )

    # --- Assert preprocessing output ---
    for p in ["p1", "p2"]:
        for stage in proc_config["stages"]:
            assert not preprodata[p][stage].empty, (
                f"Invalid Data: empty preprocessed data ({p}, {stage})"
            )

    ### =============PROCESSING=============== ###
    """Hacen comentario esta parte para que no tengan que esperar, en ese caso descomentan el import de datos procesados, y viceversa """
    # try:
    #     procdata = proc.process(
    #         preprodata=preprodata,
    #         stages=proc_config["stages"],
    #         sessions=proc_config["sessions"],
    #         min_freq=proc_config["min_freq"],
    #         max_freq=proc_config["max_freq"],
    #         fs=window_config["fs"],
    #         winsz=window_config["winsz"],
    #         ovlap=window_config["ovlap"],
    #         channels=proc_config["channels"],
    #     )
    # except Exception as e:
    #     raise RuntimeError(f"Feature Extraction Error: {str(e)}")

    # --- Assert processing output ---
    # for stage in proc_config["stages"]:
    #     assert not procdata[stage].empty, (
    #         f"Feature Extraction Error: empty procdata ({stage})"
    #     )

    ### =============SAVE INTERMEDIATE=============== ###
    procdata_path = os.path.join(cwd, "src", "C1", "outputs", "procdata.pkl")
    # with open(procdata_path, "wb") as f:
    #     pickle.dump(procdata, f)

    with open(procdata_path, "rb") as f:
        procdata = pickle.load(f)

    ### =============BAND AVERAGING=============== ###
    band_procdata = proc.get_band_average(
        procdata=procdata,
        band_freqs=proc_config["bandfreqs"],
        stages=proc_config["stages"],
        channels=proc_config["channels"],
    )

    ### =============NORMALIZATION=============== ###
    procdata_norm = proc.normalize_by_baseline(band_procdata)

    ### =============FEATURE MATRIX=============== ###
    feat_mat = proc.build_feature_matrix(procdata_norm)

    # --- Assert feature matrix ---
    assert not feat_mat.empty, "Invalid Dataset: feature matrix is empty"
    assert "Target" in feat_mat.columns, "Invalid Dataset: missing Target"

    ### =============EDA REPORT=============== ###
    try:
        eda_report = st.generate_report(feat_mat)
    except Exception as e:
        raise RuntimeError(f"Statistical Analysis Error: {str(e)}")

    # --- Assert EDA ---
    assert eda_report is not None, "Statistical Analysis Error: EDA failed"

    ### =============OUTPUTS=============== ###
    feat_mat_path = os.path.join(cwd, "src", "C1", "outputs", "feat_mat.pkl")
    with open(feat_mat_path, "wb") as f:
        pickle.dump(feat_mat, f)

    eda_report_path = os.path.join(cwd, "src", "C1", "outputs", "eda_report.pkl")
    with open(eda_report_path, "wb") as f:
        pickle.dump(eda_report, f)

    return feat_mat, eda_report
