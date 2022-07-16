#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2022 Ibuki Kuroyanai

"""Normalize wave data from download dir."""

import argparse
import logging
import glob
import os

import librosa
import numpy as np
from tqdm import tqdm
import soundfile as sf
from asd_tools.utils import write_hdf5


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features."
    )
    parser.add_argument(
        "--download_dir",
        default=None,
        type=str,
        help="Download dir or scp file.",
    )
    parser.add_argument(
        "--dumpdir", type=str, required=True, help="directory to dump feature files."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Select uav or idmt."
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.getLogger("matplotlib.font_manager").disabled = True
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    sr = 16000
    sec = 2.0
    os.makedirs(args.dumpdir, exist_ok=True)
    # process each data
    if args.dataset == "uav":
        wave_fname_list = glob.glob(f"{args.download_dir}/Dataset/**/**/**/**/*.wav")
    elif args.dataset == "idmt":
        wave_fname_list = glob.glob(f"{args.download_dir}/test_cut/**/*.wav")
        wave_fname_list += glob.glob(f"{args.download_dir}/train_cut/**/*.wav")
    elif args.dataset == "audioset":
        with open(args.download_dir, "r") as f:
            wave_fname_list = [s.strip() for s in f.readlines()]
    for fname in tqdm(sorted(wave_fname_list)):
        if args.dataset == "audioset":
            wav_id = fname.split("/")[-1].split(".")[0]
        else:
            wav_id = "-".join(fname.split("/")[2:]).split(".")[0]
        logging.info(f"load:{fname}")
        x, orig_sr = sf.read(fname)
        x = librosa.resample(x, orig_sr=orig_sr, target_sr=sr)
        save_path = os.path.join(args.dumpdir, f"{wav_id}.h5")
        x_len = len(x)
        if x_len < sr * sec:
            continue
        n_split = max(int((x_len - sr * sec) // sr), 0) + 1
        for i in range(n_split):
            start_idx = (x_len // n_split) * i
            end_idx = start_idx + int(sr * sec)
            if end_idx > x_len:
                start_idx = x_len - int(sr * sec)
                end_idx = x_len
            tmp_wave = x[start_idx:end_idx]
            if tmp_wave.std() == 0:
                logging.warning(
                    f"Std of wave{i} in {save_path} is {tmp_wave.std()}."
                    "It added gaussian noise."
                )
                tmp_wave += np.random.randn(len(tmp_wave))
            write_hdf5(
                save_path,
                f"wave{i}",
                tmp_wave.astype(np.float32),
            )


if __name__ == "__main__":
    main()
