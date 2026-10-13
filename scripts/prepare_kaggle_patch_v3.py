"""Package source only; upload this ZIP, the patch HDF5, and the notebook to Kaggle."""
from pathlib import Path
import zipfile
from utils.checkpoint import file_hash


def main():
    root=Path(__file__).resolve().parents[1]
    out=root/'artifacts/kaggle_patch_v3_source.zip'
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as archive:
        for directory in ('models','simulation','utils','scripts'):
            for file in sorted((root/directory).rglob('*.py')):archive.write(file,file.relative_to(root))
    print(out);print('SHA256',file_hash(out))


if __name__=='__main__':main()
