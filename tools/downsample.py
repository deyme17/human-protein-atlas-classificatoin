import os
import random
import argparse
import pandas as pd


def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['labels'] = df['Target'].astype(str).apply(
        lambda x: set(int(l) for l in x.strip().split())
    )
    return df


def find_candidates(df: pd.DataFrame, target_classes: set, mode: str = "all") -> pd.DataFrame:
    if mode == 'only':
        mask = df['labels'].apply(lambda lbls: len(lbls) == 1 and lbls.issubset(target_classes))
    else:
        mask = df['labels'].apply(lambda lbls: lbls.issubset(target_classes) and len(lbls) > 0)
    return df[mask].copy()


def resolve_paths(candidates: pd.DataFrame, directories: list[str]) -> list[tuple[str, str]]:
    """
    For each candidate row, find the actual file on disk across given directories.
    Returns list of (index, filepath) for files that exist.
    """
    found = []
    for _, row in candidates.iterrows():
        for d in directories:
            path = os.path.join(d, row['Id'] + '.png')
            if os.path.exists(path):
                found.append((row['Id'], path))
                break
    return found


def print_breakdown(candidates: pd.DataFrame, target_classes: set) -> None:
    print()
    total = len(candidates)

    for r in range(1, len(target_classes) + 1):
        from itertools import combinations
        for combo in combinations(sorted(target_classes), r):
            combo_set = set(combo)
            count = candidates['labels'].apply(lambda l: l == combo_set).sum()
            label = ' + '.join(f'class {c}' for c in combo)
            print(f"  only {label:<30}: {count}")

    print(f"  {'total candidates':<34}: {total}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class-ids', nargs='+', type=int, required=True)
    parser.add_argument('--train_csv', default='data/train.csv', type=str)
    parser.add_argument('--external_csv', default='data/external.csv', type=str)
    parser.add_argument('--dirs', nargs='+', required=True,
        help='Directories to search for images, e.g. --dirs data/train data/external'
    )
    parser.add_argument('--mode', default='all', choices=['all', 'only'],
        help="'only' = remove images with a single target class label; 'all' = include mixed target-class combos (default)"
    )
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--dry_run', action='store_true', help='Preview without deleting')
    args = parser.parse_args()

    # load
    dfs = []
    if os.path.exists(args.train_csv):
        dfs.append(load_csv(args.train_csv))
    if os.path.exists(args.external_csv):
        dfs.append(load_csv(args.external_csv))

    if not dfs:
        print("No CSV files found.")
        return

    df = pd.concat(dfs, ignore_index=True)
    target_classes = set(args.class_ids)

    # find candidates
    candidates = find_candidates(df, target_classes, args.mode)

    # resolve to actual files
    found = resolve_paths(candidates, args.dirs)
    print(f"\nFiles found on disk : {len(found)} / {len(candidates)}")
    if not found:
        print("No files found on disk. Check --dirs paths.")
        return

    found_ids = {fid for fid, _ in found}
    candidates = candidates[candidates['Id'].isin(found_ids)]

    print("\nCandidate images (labels are subset of target_classes):")
    print_breakdown(candidates, target_classes)

    # fraction
    while True:
        try:
            fraction = float(input("\nFraction to remove (0.0 - 1.0): "))
            if 0.0 < fraction <= 1.0:
                break
            print("Must be between 0 and 1.")
        except ValueError:
            print("Enter a number.")

    random.seed(args.seed)
    random.shuffle(found)
    n_remove = int(len(found) * fraction)
    to_delete = found[:n_remove]

    print(f"\nWill delete {n_remove} / {len(found)} files ({fraction*100:.1f}%)")

    if args.dry_run:
        print("\nDry run — sample of files that would be deleted:")
        for _, path in to_delete[:10]:
            print(f"  {path}")
        if len(to_delete) > 10:
            print(f"  ... and {len(to_delete) - 10} more")
        print("\nRemove --dry_run to apply.")
        return

    confirm = input(f"\nDelete {n_remove} files? [y/N]: ")
    if confirm.lower() != 'y':
        print("Aborted.")
        return

    deleted, missing = 0, 0
    for _, path in to_delete:
        if os.path.exists(path):
            os.remove(path)
            deleted += 1
        else:
            missing += 1

    print(f"\nDeleted : {deleted}")
    if missing:
        print(f"Missing : {missing} (already gone)")


if __name__ == '__main__':
    main()