from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import compute_hashes, DSU
from sklearn.neighbors import BallTree
from config import PathConfig, DataConfig
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np



def build_hash_df(image_ids: list[str], 
                  train_root: Path = PathConfig.train_dir, 
                  external_root: Path = PathConfig.external_dir) -> pd.DataFrame:
    
    def resolve_root(img_id):
        return train_root if '-' in img_id else external_root
    
    def process_one(img_id):
        path = resolve_root(img_id) / f"{img_id}.png"
        if not path.exists():
            print(f"[WARN] Not found: {path}")
            return None
        try:
            md5, phash = compute_hashes(path)
            return {"Id": img_id, "hash": md5, "phash": phash}
        except Exception as e:
            print(f"[WARN] Failed on {img_id}: {e}")
            return None
        
    records = []
    with ThreadPoolExecutor(max_workers=DataConfig.n_workers) as ex:
        futures = [ex.submit(process_one, img_id) for img_id in image_ids]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Hashing"):
            res = f.result()
            if res is not None:
                records.append(res)

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
    if len(hash_df) == 0:
        hash_df["group_id"] = []
        return hash_df

    # unpackbits optimization
    hashes = hash_df["phash"].values.astype(np.uint64)
    hashes_bytes = hashes.byteswap().view(np.uint8).reshape(-1, 8)
    hashes_matrix = np.unpackbits(hashes_bytes, axis=1)
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