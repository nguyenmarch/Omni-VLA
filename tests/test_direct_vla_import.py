import zipfile
import pytest
from scripts.import_direct_vla import import_candidate


@pytest.mark.parametrize('names', [['../best.pt','last.pt','metrics.jsonl'],
                                  ['best.pt','best.pt','metrics.jsonl']])
def test_import_rejects_unsafe_or_duplicate_archive_before_creating_outputs(tmp_path,names):
    archive=tmp_path/'outputs.zip'
    with zipfile.ZipFile(archive,'w') as file:
        for name in names:
            file.writestr(name,b'invalid')
    output=tmp_path/'candidate'
    with pytest.raises(ValueError,match='Expected exactly'):
        import_candidate(archive,tmp_path/'missing.h5',output,tmp_path/'report.json')
    assert not output.exists()
