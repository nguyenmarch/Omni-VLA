"""Create source ZIP excluding local datasets, weights, secrets and environments."""
from pathlib import Path
import argparse
import zipfile

def package(output):
    root=Path(__file__).resolve().parents[1];output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        for folder in ('configs','models','simulation','scripts','utils','docs','tests','notebooks'):
            for path in (root/folder).rglob('*'):
                if (path.is_file() and '__pycache__' not in path.parts
                        and path.relative_to(root).as_posix() != 'docs/policies/DIRECT_VLA_HANDOFF.json'):
                    archive.write(path,Path('omni_vla')/path.relative_to(root))
        for name in ('README.md','pyproject.toml'):
            archive.write(root/name,Path('omni_vla')/name)
    print(output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='artifacts/kaggle_source.zip');package(p.parse_args().output)
