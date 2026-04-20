from utils import compute_hash, compute_phash, DSU
from sklearn.neighbors import BallTree
from config import PathConfig
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np



def build_hash_df(image_ids: list[str], root: Path = PathConfig.train_dir) -> pd.DataFrame:
    records = []
    for img_id in tqdm(image_ids, desc="Hashing"):
        records.append({
            "Id": img_id,
            "hash": compute_hash(img_id, root),
            "phash": compute_phash(img_id, root),
        })
    return pd.DataFrame(records)



def remove_dublicates(hash_df: pd.DataFrame) -> pd.DataFrame:
    df = (hash_df.sort_values("Id")
                 .drop_duplicates(subset=["hash"], keep="first")
                 .reset_index(drop=True))
    removed = len(hash_df) - len(df)
    print(f"Dublicates removed: {removed}")
    return df



def assign_groups(hash_df: pd.DataFrame, threshold: int = 8) -> pd.DataFrame:
    """
    Find groups (near-dublicates) using DSU and hamming distance.
    Args:
        hash_df (pd.DataFrame): dataframe with "Id" and "phash" columns.
        threshold (int): threshold for assigning imgs to groups (by hamming distance).
    Rerturn:
        hash_df with new "group_id" column.
    """
    hashes_matrix = np.array([h.hash.flatten() for h in hash_df["phash"]]).astype(int)
    ids = hash_df["Id"].tolist()
    
    # build tree with metric='hamming' for a fast search
    tree = BallTree(hashes_matrix, metric='hamming')
    
    # find all data points within a specified distance
    radius = threshold / 64.0
    indices = tree.query_radius(hashes_matrix, r=radius)
    
    # union groups
    dsu = DSU(ids)
    for i, matches in enumerate(indices):
        for m_idx in matches:
            if i < m_idx:
                dsu.union(ids[i], ids[m_idx])

    # root to group id
    roots = [dsu.find(i) for i in ids]
    root_to_int = {r: idx for idx, r in enumerate(dict.fromkeys(roots))}
    hash_df["group_id"] = [root_to_int[r] for r in roots]

    # report
    n_groups = len(set(roots))
    print(f"Groups Assigned: {n_groups}")
    print(f"Near-duplicates: {len(hash_df) / n_groups:.2f}")
    group_sizes = hash_df["group_id"].value_counts()
    print(f"\nGroup size distribution:")
    print(group_sizes.value_counts().sort_index().rename("# groups").to_string())
    
    return hash_df



def assert_no_leakage(df: pd.DataFrame, train_ids: list[str], val_ids: list[str]):
    id_to_group = df.set_index("Id")["group_id"].to_dict()
    train_groups = {id_to_group[i] for i in train_ids}
    val_groups = {id_to_group[i] for i in val_ids}
    leaked = train_groups & val_groups
    assert len(leaked) == 0, (
        f"DATA LEAKAGE: {len(leaked)} group(s) appear in both splits."
    )