from __future__ import absolute_import, division, print_function

import os
import argparse
import glob

import tqdm
import imageio.v2 as imageio
import numpy as np


def load_channel(path):
    img = imageio.imread(path)
    return img[:, :, 0] if img.ndim == 3 else img


def load_ids(input_dir):
    red_filenames = glob.glob(os.path.join(input_dir, '*_red.png'))
    return [f[:-8] for f in red_filenames]


def process(id_lists, output_dir):
    for id_str in tqdm.tqdm(id_lists):
        output_filename = os.path.join(output_dir, os.path.basename(id_str) + '.png')
        if os.path.exists(output_filename):
            continue

        channels = {c: id_str + f'_{c}.png' for c in ('red', 'green', 'blue', 'yellow')}
        if not all(os.path.exists(p) for p in channels.values()):
            continue

        stacked = np.stack(
            [load_channel(p) for p in channels.values()], axis=2
        ).astype(np.uint8)

        imageio.imwrite(output_filename, stacked)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    process(load_ids(args.input_dir), args.output_dir)


if __name__ == '__main__':
    main()