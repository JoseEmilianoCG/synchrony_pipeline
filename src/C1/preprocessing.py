import pandas as pd
import scipy as sp
import json
import os
import glob
### =====================IMPORTS===================== ###

## Find current path

cwd = os.getcwd()

## Import filter_params

filter_params_path = os.path.join(cwd, "src", "C1", "inputs", "filter_params.json")
with open(filter_params_path, "r") as f:
    filter_params = json.load(f)

### =====================UTILS===================== ###

## Data partition by markers

def segment_by_markers(data, markers):
    """
    Inputs:
    data = Df of raw unpartitioned data
    markers = Df containing markers, both timestamp and

    Outputs:
    preprodata = Dict of dfs containing the preprocessed data

    """

    raw = data.copy().drop("Unnamed: 0", axis=1).dropna()
    markr = markers.copy()

    # The raw data has has duplicates of the headers because of concatenation, here we simply fix this
    raw = raw[raw["TP9"] != "TP9"]
    raw = raw.astype(float).reset_index(drop=True)

    segments = {}

    # Iterate through markers to define segments
    for i in range(len(markers)):
        start_ts = float(markr.loc[i, "unix_ts"])
        label = markr.loc[i, "label"].replace("_start", "")

        # Define end timestamp (next marker or end of raw data)
        if i < len(markr) - 1:
            end_ts = float(markr.loc[i + 1, "unix_ts"])
        else:
            end_ts = float(raw["unix_ts"].max())

        # Extract segment based on time window
        segment = raw[(raw["unix_ts"] >= start_ts) & (raw["unix_ts"] < end_ts)].copy()

        # Store directly (one segment per label)
        segments[label] = segment.drop(["board_ts", "unix_ts"], axis=1)

    return segments


## Digital filters

# Filter object creation

SOS = sp.signal.butter(
    filter_params["filter_order"],
    filter_params["bandwidth"],
    btype=filter_params["type"],
    fs=filter_params["fs"],
    output=filter_params["output"],
)


# Filtering function
def digfilter(data):
    """
    Inputs:
    data = Df containing raw data

    Outputs:
    data_f = Df of filtered data

    """

    data_f = data.copy()
    data_f[data.columns] = sp.signal.sosfiltfilt(SOS, data.values, axis=0)

    return data_f


### =====================PREPROCESSING FUNCTION===================== ###


def prepro(
    stages,
    sessions,
    cwd,
    baseline_slice=slice(2561, 25601),
    task_slice=slice(2561, 64001),
):
    """
    Inputs:
    stages = List of stages
    sessions = List of sessions
    folderpath = Path for rawdata folder
    baseline_slice = Slice indexes for baseline data
    task_slice = Slice indexes for task data

    Outputs:
    preprodata = Dict of dfs containing the preprocessed data

    """
    # Preprocessed data structure
    preprodata = {
        "p1": {s: pd.DataFrame() for s in stages},
        "p2": {s: pd.DataFrame() for s in stages},
    }

    for session_id in sessions:
        # Data import of corresponding session
        pattern = f"S{session_id}R*"
        base_path = os.path.join(cwd, "src", "C1", "inputs", "rawdata")
        snpath = glob.glob(os.path.join(base_path, pattern))[0]

        p1raw = pd.read_csv(os.path.join(snpath, "Raw.csv"))

        p2raw = pd.read_csv(os.path.join(snpath, "Raw2.csv"))

        markers = pd.read_csv(os.path.join(snpath, "markers.csv"))

        # Data partition
        p1parts = segment_by_markers(p1raw, markers)
        p2parts = segment_by_markers(p2raw, markers)

        for stage in stages:
            for p, raw in zip(["p1", "p2"], [p1parts, p2parts]):
                # Data filtering
                stage_df = digfilter(raw[stage])
                # Data slicing
                stage_df = stage_df.iloc[
                    baseline_slice if stage == "baseline" else task_slice, :
                ]

                # Common mean reference
                stage_df = stage_df.subtract(stage_df.mean(axis=1), axis=0)
                stage_df.insert(0, "Session", session_id)

                # Preprocessed data is integrated
                preprodata[p][stage] = pd.concat(
                    [preprodata[p][stage], stage_df], ignore_index=True
                )

    return preprodata
