# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: MIT
#
# SPDX-FileContributor: Anjith George <anjith.george@idiap.ch>

import os
import shutil
import glob
import numpy as np
import pandas as pd
import h5py
import cv2

# ---------------------------------------------------------------- config
PROTOCOL_ROOT = 'Protocols'  
OUT_ROOT      = 'CROPPED/'   

DATABASES = {
    'cbsr-nir-vis2': {
        'crop_root': '/cbsr-nir-vis2/CROPS',
        'h5_root':   'cbsr-nir-vis2/preprocessed/',
    },
    'polathermal': {
        'crop_root': '/polathermal/CROPS',
        'h5_root':   'polathermal/preprocessed/',
    },
    'mcxface': {
        'crop_root': '/mcxface/CROPS',
        'h5_root':   'mcxface/preprocessed/',
    },
}


# ---------------------------------------------------------------- helpers
def read_h5_image(fullpath):
    with h5py.File(fullpath, 'r') as f:
        img = f['array'][:]
    img = np.transpose(img, (1, 2, 0)).astype('uint8')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def materialize(spath, cfg, dst):
    """Put the image for `spath` at `dst`. Returns 'copied', 'decoded' or None."""
    existing = os.path.join(cfg['crop_root'], spath)
    if os.path.exists(existing):
        shutil.copy(existing, dst)
        return 'copied'

    if cfg['h5_root'] is not None:
        src_h5 = os.path.join(cfg['h5_root'], os.path.splitext(spath)[0] + '.h5')
        if os.path.exists(src_h5):
            img = read_h5_image(src_h5)
            if not cv2.imwrite(dst, img):
                raise IOError(f'failed to write {dst}')
            return 'decoded'

    return None


# ---------------------------------------------------------------- run
for db, cfg in DATABASES.items():
    proto_dir = os.path.join(PROTOCOL_ROOT, db, 'Protocols')
    crop_out  = os.path.join(OUT_ROOT, db, 'CROPS')

    csvs = sorted(glob.glob(os.path.join(proto_dir, '*.csv')))
    if not csvs:
        print(f'=== {db} === no csv under {proto_dir}, skipping')
        continue

    os.makedirs(crop_out, exist_ok=True)
    print(f'\n=== {db} === {len(csvs)} csv files')

    wanted = set()
    for path in csvs:
        df = pd.read_csv(path)
        wanted.update(df['SPATH'].astype(str))
    print(f'  {len(wanted)} unique images referenced')

    copied = decoded = present = 0
    missing = []

    for spath in sorted(wanted):
        dst = os.path.join(crop_out, spath)
        if os.path.exists(dst):
            present += 1
            continue

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        result = materialize(spath, cfg, dst)

        if result == 'copied':
            copied += 1
        elif result == 'decoded':
            decoded += 1
        else:
            missing.append(spath)

    print(f'  {copied} copied, {decoded} decoded from h5, '
          f'{present} already present, {len(missing)} missing')

    if missing:
        log = os.path.join(OUT_ROOT, db, 'missing.txt')
        with open(log, 'w') as f:
            f.write('\n'.join(missing))
        print(f'  !! could not resolve {len(missing)} images, see {log}')

    # verify the tree matches the protocol files exactly
    on_disk = set()
    for root, _, files in os.walk(crop_out):
        for fn in files:
            on_disk.add(os.path.relpath(os.path.join(root, fn), crop_out))

    extra = on_disk - wanted
    print(f'  tree: {len(on_disk)} files on disk, '
          f'{len(wanted - on_disk)} referenced but absent, {len(extra)} extra')