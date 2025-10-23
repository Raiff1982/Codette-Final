import asyncio
from pathlib import Path
from codette_quantum_multicore2 import analyse_cocoons_async, load_cocoon

def test_async_analysis():
    cocoon_dir = Path('.')
    result = asyncio.run(analyse_cocoons_async(cocoon_dir))
    assert isinstance(result, list)


def test_invalid_cocoon_returns_none(tmp_path):
    bad_file = tmp_path / "bad.cocoon"
    bad_file.write_text("{not json}")
    assert load_cocoon(bad_file) is None
