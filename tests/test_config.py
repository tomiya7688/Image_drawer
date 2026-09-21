from image_drawer.config import CONFIG_ENV_VAR, load_config


def test_load_config_from_explicit_path(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[project]\nname = "test-project"\n[logging]\nlevel = "DEBUG"\n',
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["project"]["name"] == "test-project"
    assert config["logging"]["level"] == "DEBUG"


def test_load_config_from_environment(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text('[project]\nname = "env-project"\n', encoding="utf-8")
    monkeypatch.setenv(CONFIG_ENV_VAR, str(config_path))

    assert load_config()["project"]["name"] == "env-project"


def test_load_config_without_path_or_environment(monkeypatch):
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
    assert load_config() == {}
