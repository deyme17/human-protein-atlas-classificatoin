import pandas as pd

from dataset import save_dataset, get_dataset
from preproc import (
    build_hash_df, remove_dublicates, assign_groups, group_stratified_split
)
from config import PathConfig, DataConfig
from utils import prepare_train_data



def create_dataset():
    # load labels
    print("Loading labels ...")
    train_df = pd.read_csv(PathConfig.train_labels_path)
    external_df = pd.read_csv(PathConfig.external_labels_path)

    # concat train and external
    combined_df = pd.concat([train_df, external_df], ignore_index=True)
    print(f"\ttrain: {len(train_df):>6} samples")
    print(f"\texternal: {len(external_df):>6} samples")
    print(f"\ttotal: {len(combined_df):>6} samples")

    # prepare data
    print("\nEncoding labels ...")
    _, all_labels = prepare_train_data(combined_df)

    # hash_df
    print("\nBuilding hash index ...")
    hash_df = build_hash_df(combined_df["Id"].tolist())

    # remove dublications
    print("\nRemoving exact duplicates ...")
    hash_df = remove_dublicates(hash_df)

    # assign groups
    print("\nAssigning near-duplicate groups ...")
    hash_df = assign_groups(hash_df)

    # update combined_df and labels
    dedup_ids = set(hash_df["Id"])
    original_ids = pd.concat([train_df, external_df], ignore_index=True)["Id"].tolist()
    all_labels = [lbl for img_id, lbl in zip(original_ids, all_labels) if img_id in dedup_ids]
    combined_df = combined_df[combined_df["Id"].isin(dedup_ids)].reset_index(drop=True)

    print(f"\nAfter dedup: {len(combined_df)} samples")

    # stratified group split
    print("\nSplitting into train / val ...")
    train_ids, train_labels, val_ids, val_labels = group_stratified_split(
        hash_df,
        all_labels,
        val_fraction=DataConfig.val_fraction,
    )
    print(f"\ttrain: {len(train_ids):>6} samples")
    print(f"\tval: {len(val_ids):>6} samples")

    # build dataset
    print("\nBuilding datasets ...")
    train_ds = get_dataset(image_ids=train_ids, labels=train_labels)
    valid_ds = get_dataset(image_ids=val_ids, labels=val_labels)

    # test set
    sub_df = pd.read_csv(PathConfig.sample_submission_path)
    test_ids = sub_df["Id"].tolist()
    test_ds = get_dataset(image_ids=test_ids, labels=None)
    print(f"\ttest: {len(test_ids):>6} samples")

    # save
    print("\nSaving datasets ...")
    save_dataset(train_ds, PathConfig.train_ds_path)
    print(f"\ttrain saved -> {PathConfig.train_ds_path}")

    save_dataset(valid_ds, PathConfig.valid_ds_path)
    print(f"\tval saved -> {PathConfig.valid_ds_path}")

    save_dataset(test_ds, PathConfig.test_ds_path)
    print(f"\ttest saved -> {PathConfig.test_ds_path}")

    print("\nDone.")


if __name__ == "__main__":
    create_dataset()