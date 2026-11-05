from config.settings import BASE_DIR, MODEL_DIR


def test_project_setup():
    assert BASE_DIR.exists()
    assert MODEL_DIR.exists()