import os
import sys
import tempfile
from pathlib import Path

import pytest

_tmp_db = Path(tempfile.gettempdir()) / "dwarpal_test.db"
if _tmp_db.exists():
    _tmp_db.unlink()
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp_db}"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from app.db import Base, engine
    from app.main import app

    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    with TestClient(app) as c:
        yield c
