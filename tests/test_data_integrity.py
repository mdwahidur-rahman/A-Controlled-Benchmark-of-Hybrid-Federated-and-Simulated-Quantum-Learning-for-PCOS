"""Checks on the shipped manifest and split files (metadata only; no images)."""
import pathlib, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
man = pd.read_csv(ROOT / "data" / "manifest.csv"); spl = pd.read_csv(ROOT / "data" / "splits.csv")

def test_counts():
    assert len(man) == 11784 and man.label.sum() == 6784 and (man.label == 0).sum() == 5000
    assert man.md5.nunique() == 3996 and man.g0.nunique() == 3255

def test_groups_are_class_pure():
    assert (man.groupby("g0").label.nunique() == 1).all()

def test_exact_duplicates_share_a_group():
    assert (man.groupby("md5").g0.nunique() == 1).all()

def test_group_splits_are_disjoint_and_proportional():
    for s in range(5):
        part = spl[f"grp_s{s}"]; assert set(part.unique()) == {"train", "val", "test"}
        assert (man.assign(p=part).groupby("g0").p.nunique() == 1).all(), "a g0 group straddles partitions"
        fr = part.value_counts(normalize=True); assert abs(fr["train"] - 0.64) < 0.01 and abs(fr["val"] - 0.16) < 0.01 and abs(fr["test"] - 0.20) < 0.01

def test_image_level_split_is_not_group_disjoint():
    straddling = sum((man.assign(p=spl[f"img_s{s}"]).groupby("g0").p.nunique() > 1).sum() for s in range(5)); assert straddling > 0
