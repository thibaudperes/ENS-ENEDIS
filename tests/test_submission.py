"""Tests du format de soumission."""

import numpy as np
import pandas as pd
import pytest

from enedis_imputation.submission import (
    SAMPLE_PATH,
    SUBMISSION_PATH,
    check_submission,
    read_sample,
    to_submission,
)

STAMPS = ["2023-01-09 00:00:00", "2023-01-09 00:30:00", "2023-01-09 01:00:00"]


def _sample() -> pd.DataFrame:
    return pd.DataFrame(
        {"holed_1": [1.0, 2.0, 3.0], "holed_2": [4.0, 5.0, 6.0]},
        index=pd.Index(STAMPS, name="Horodate"),
    )


def _holed() -> pd.DataFrame:
    return pd.DataFrame(
        {"holed_2": [7.0, np.nan, 9.0], "holed_1": [np.nan, 2.5, 3.5]},
        index=pd.DatetimeIndex(pd.to_datetime(STAMPS), name="Horodate"),
    )


def test_to_submission_matches_sample_format() -> None:
    """Mêmes colonnes dans le même ordre, même index texte que le fichier exemple."""
    filled = _holed().fillna(1.0)
    sub = to_submission(filled, _sample())
    assert list(sub.columns) == ["holed_1", "holed_2"]
    assert list(sub.index) == STAMPS
    check_submission(sub, _sample(), _holed())


def test_check_submission_rejects_errors() -> None:
    """NaN, valeur négative ou valeur observée modifiée sont refusés."""
    sample, holed = _sample(), _holed()
    good = to_submission(holed.fillna(1.0), sample)
    for broken, message in [
        (good.mask(good == 1.0), "manquante"),
        (good.where(good != 1.0, -1.0), "négative"),
        (good.assign(holed_2=good["holed_2"] + 1), "observées"),
        (good[["holed_2", "holed_1"]], "Colonnes"),
    ]:
        with pytest.raises(ValueError, match=message):
            check_submission(broken, sample, holed)


@pytest.mark.skipif(not SUBMISSION_PATH.exists(), reason="soumission pas encore produite")
def test_written_submission_has_the_exact_format() -> None:
    """Le fichier produit a exactement l'en-tête, l'index et la forme du fichier exemple."""
    sample = read_sample(SAMPLE_PATH)
    sub = read_sample(SUBMISSION_PATH)
    assert sub.columns.equals(sample.columns)
    assert sub.index.equals(sample.index)
    assert sub.shape == (1057, 1000)
    values = sub.to_numpy()
    assert not np.isnan(values).any() and (values >= 0).all()
    with open(SAMPLE_PATH, "rb") as f_sample, open(SUBMISSION_PATH, "rb") as f_sub:
        # même en-tête octet pour octet, fins de ligne comprises (\r\n dans le fichier exemple)
        assert f_sample.readline() == f_sub.readline()
