"""Batch-run the AR overlay on all chessboard images and produce a CSV/JSON report.

This imports the existing `phase2_ar_overlay` module and calls its `run()`
function for each image while capturing the printed reprojection statistics.
Overlays are written to `results/ar_visuals/` and a `report.csv` / `report.json`
are created alongside them.
"""
import os
import glob
import argparse
import io
import json
import csv
from contextlib import redirect_stdout

import sys
_THIS_DIR = os.path.dirname(__file__)
# ensure the phase_2 directory is on sys.path so we can import phase2_ar_overlay
sys.path.insert(0, os.path.abspath(_THIS_DIR))
import phase2_ar_overlay as ar


def discover_images(images_dir):
    pattern = os.path.join(images_dir, 'calibrated_chessboard_*.jpg')
    files = sorted(glob.glob(pattern))
    # filter out visual preview images that include '_visual' in name
    files = [f for f in files if '_visual' not in os.path.basename(f)]
    return files


def run_batch(images_dir, out_dir, calibration_npz, board_cols, board_rows, square_size):
    images = discover_images(images_dir)
    os.makedirs(out_dir, exist_ok=True)
    report = []

    for img_path in images:
        name = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(out_dir, f'ar_overlay_{name}.jpg')

        args = argparse.Namespace(
            image=img_path,
            calibration=calibration_npz,
            square_size=square_size,
            board_cols=board_cols,
            board_rows=board_rows,
            out=out_path
        )

        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                ret = ar.run(args)
            except Exception as e:
                ret = 1
                print('ERROR running on', img_path, e)

        output = buf.getvalue()

        entry = {
            'image': img_path,
            'out': out_path,
            'status': 'ok' if ret == 0 else 'error',
            'mean_px': None,
            'max_px': None,
            'raw_output': output.strip()
        }

        # parse reprojection line if present
        for line in output.splitlines():
            if 'Reprojection error' in line:
                # format: Reprojection error (mean px): 0.835, max px: 3.561
                try:
                    parts = line.split(':')[-1].strip()
                    mean_part, max_part = parts.split(',')
                    mean_px = float(mean_part.replace('mean px', '').strip())
                    max_px = float(max_part.replace('max px', '').strip())
                    entry['mean_px'] = mean_px
                    entry['max_px'] = max_px
                except Exception:
                    pass

        report.append(entry)

    # write JSON and CSV
    json_path = os.path.join(out_dir, 'ar_report.json')
    csv_path = os.path.join(out_dir, 'ar_report.csv')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'out', 'status', 'mean_px', 'max_px'])
        for e in report:
            writer.writerow([e['image'], e['out'], e['status'], e['mean_px'], e['max_px']])

    print('Wrote report:', json_path, csv_path)
    return report


def main():
    parser = argparse.ArgumentParser()
    # use the calibrated images folder under phase_1/results
    parser.add_argument('--images-dir', default=os.path.join('..', 'phase_1', 'results', 'calibrated images'))
    parser.add_argument('--out-dir', default=os.path.join('results', 'ar_visuals'))
    parser.add_argument('--calibration', default=os.path.join('..', 'phase_1', 'results', 'calibration_results.npz'))
    parser.add_argument('--board-cols', type=int, default=7)
    parser.add_argument('--board-rows', type=int, default=6)
    parser.add_argument('--square-size', type=float, default=0.025)
    args = parser.parse_args()

    images_dir = os.path.abspath(args.images_dir)
    out_dir = os.path.abspath(args.out_dir)
    calibration = os.path.abspath(args.calibration)

    run_batch(images_dir, out_dir, calibration, args.board_cols, args.board_rows, args.square_size)


if __name__ == '__main__':
    main()
