#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2022 Ibuki Kuroyanai

"""Normalize wave data from download dir."""

import argparse
import logging
import os
import json

import librosa
import numpy as np
from tqdm import tqdm

from serial_oe.utils import write_hdf5


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
        "--statistic_path", type=str, default="", help="wave statistic in json file."
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
    if not os.path.isfile(args.statistic_path):
        wave_fname_list = os.listdir(args.download_dir)
        # wave_fname_list = [fname for fname in wave_fname_list if "target" not in fname]
        wave, _ = librosa.load(
            os.path.join(args.download_dir, wave_fname_list[0]), sr=sr
        )
        tmp = np.zeros((len(wave_fname_list), len(wave)))
        for i, fname in enumerate(tqdm(wave_fname_list)):
            logging.info(f"fname:{fname}")
            tmp[i], _ = librosa.load(os.path.join(args.download_dir, fname), sr=sr)
        statistic = {}
        statistic["mean"] = tmp.mean()
        statistic["std"] = tmp.std()
        with open(args.statistic_path, "w") as f:
            json.dump(statistic, f)
        logging.info(f"Successfully saved statistic to {args.statistic_path}.")

    # process each data
    if args.download_dir.endswith(".scp"):
        with open(args.download_dir, "r") as f:
            wave_fname_list = [s.strip() for s in f.readlines()]
    else:
        wave_fname_list = os.listdir(args.download_dir)

    for fname in tqdm(wave_fname_list):
        if args.download_dir.endswith(".scp"):
            wav_id = fname.split("/")[-1].split(".")[0]
            path = fname
            if os.path.getsize(path) < 500000:
                logging.info(f"Size of {path} is too low!")
                continue
        else:
            wav_id = fname.split(".")[0]
            path = os.path.join(args.download_dir, fname)
        logging.info(f"load:{path}")
        x, _ = librosa.load(path=path, sr=sr)
        save_path = os.path.join(args.dumpdir, f"{wav_id}.h5")
        if args.download_dir.endswith(".scp"):
            x_len = len(x) - int(sr * sec)
            for i in range(5):
                # if os.path.isfile(save_path):
                #     hdf5_file = h5py.File(save_path, "r")
                #     if f"wave{i}" in hdf5_file:
                #         logging.warning(f"wave{i} has already existed in {save_path}.")
                #         continue
                start_idx = (x_len // 5) * i
                end_idx = start_idx + int(sr * sec)
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
        else:
            write_hdf5(
                save_path,
                "wave",
                x.astype(np.float32),
            )


if __name__ == "__main__":
    main()
