"""Check expected dataset directories and produce a readiness report without downloading data."""
from pathlib import Path
import argparse, json


def _first_existing(paths):
    return next((p for p in paths if p.exists()), paths[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    a = ap.parse_args()
    root = Path(a.root).resolve()
    xbd_train_split = _first_existing([root/'data/xbd/train', root/'data/xbd/tier1'])
    checks = {
        'oscd_images': root/'data/oscd/Onera Satellite Change Detection dataset - Images',
        'oscd_train_labels': root/'data/oscd/Onera Satellite Change Detection dataset - Train Labels',
        'oscd_test_labels': root/'data/oscd/Onera Satellite Change Detection dataset - Test Labels',
        'sen1floods11_s1': root/'data/sen1floods11/data/flood_events/HandLabeled/S1Hand',
        'sen1floods11_labels': root/'data/sen1floods11/data/flood_events/HandLabeled/LabelHand',
        'sen1floods11_splits': root/'data/sen1floods11/splits/flood_handlabeled',
        'xbd_train_or_tier1_images': xbd_train_split/'images',
        'xbd_train_or_tier1_labels_json': xbd_train_split/'labels',
        'xbd_train_or_tier1_targets_generated': xbd_train_split/'targets',
        'xbd_labelled_test_images': root/'data/xbd/test/images',
        'xbd_labelled_test_labels_json': root/'data/xbd/test/labels',
        'xbd_labelled_test_targets_generated': root/'data/xbd/test/targets',
        'xbd_tier3_images': root/'data/xbd/tier3/images',
        'xbd_tier3_labels_json': root/'data/xbd/tier3/labels',
    }
    out = {k: {'path': str(p), 'exists': p.exists(),
                'files': sum(1 for f in p.rglob('*') if f.is_file()) if p.exists() else 0}
           for k, p in checks.items()}
    out['xbd_active_training_split'] = {'path': str(xbd_train_split), 'selected_from': ['train', 'tier1'],
                                        'note': 'The real xBD release uses tier1 in some extracts; this path is an alias, not proof labels are present.'}
    report = {'root': str(root), 'datasets': out,
              'note': 'Readiness only. No data were downloaded. Missing paths must be prepared from cited dataset sources.'}
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
