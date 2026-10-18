import numpy as np
import pandas as pd

from src import config, data

RAW = """63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0
67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,3.0,3.0,2
67.0,1.0,4.0,120.0,229.0,0.0,2.0,129.0,1.0,2.6,2.0,2.0,7.0,1
37.0,1.0,3.0,130.0,250.0,0.0,0.0,187.0,0.0,3.5,3.0,0.0,3.0,0
41.0,0.0,2.0,130.0,204.0,0.0,2.0,172.0,0.0,1.4,1.0,?,3.0,0
56.0,1.0,2.0,120.0,236.0,0.0,0.0,178.0,0.0,0.8,1.0,0.0,?,4
56.0,1.0,2.0,120.0,236.0,0.0,0.0,178.0,0.0,0.8,1.0,0.0,?,4
"""


def raw(tmp_path):
    p = tmp_path / "raw.data"
    p.write_text(RAW)
    return data.load_raw_data(p)


def test_load_and_clean(tmp_path):
    df = raw(tmp_path)
    assert df["ca"].isna().sum() == 1 and df["thal"].isna().sum() == 2
    clean, rep = data.clean_data(df)
    assert "num" not in clean.columns
    assert clean[config.TARGET].tolist() == [0, 1, 1, 0, 0, 1]
    assert rep["duplicates_removed"] == 1 and rep["rows_out"] == 6


def test_bad_values_become_nan(tmp_path):
    df = raw(tmp_path)
    df.loc[0, "chol"] = 0
    df.loc[1, "cp"] = 9
    clean, rep = data.clean_data(df)
    assert np.isnan(clean.loc[0, "chol"])
    assert pd.isna(clean.loc[1, "cp"])
    assert rep["out_of_range_set_to_nan"] == {"chol": 1}


def test_real_dataset():
    df = pd.read_csv(config.CLEAN_DATA_FILE)
    assert df.shape == (303, 14)
    X, y = data.split_features_target(df)
    assert list(X.columns) == config.FEATURE_ORDER
    assert 0.4 < y.mean() < 0.5
