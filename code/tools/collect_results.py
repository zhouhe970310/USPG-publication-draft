from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main():
    root = Path.cwd()
    output = root / 'USPG_results_for_review.zip'
    roots = ['audit', 'splits', 'results_clean', 'results_sensitivity', 'logs_clean']
    with ZipFile(output, 'w', compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for name in roots:
            path = root / name
            if path.exists():
                for file in sorted(path.rglob('*')):
                    if file.is_file():
                        archive.write(file, file.relative_to(root))
        run_files = sorted((root / 'runs').glob('*/seed*/*')) \
            if (root / 'runs').exists() else []
        for file in run_files:
            if file.is_file() and file.name in {'config.json', 'training_history.csv'}:
                archive.write(file, file.relative_to(root))
    print(output.resolve())


if __name__ == '__main__':
    main()
