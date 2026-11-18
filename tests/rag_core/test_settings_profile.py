"""The settings profile layer: a profile is read from its file only, validated strictly, and no overlay, environment variable
or secret file may set a profile key."""

import pytest
from pydantic import Field
from rag_core.settings import OverlaySection, ProfileModel, SettingsError, load, load_profile


class Chunk(ProfileModel):
    mode: str
    parent_chars: int = Field(gt=0)


class Profile(ProfileModel):
    chunking: Chunk


class Net(OverlaySection):
    url: str = "http://x"


class Overlay(OverlaySection):
    net: Net = Net()


def test_a_profile_file_is_validated_strictly(tmp_path):
    good = tmp_path / "p.yaml"
    good.write_text("chunking: {mode: structure, parent_chars: 2000}\n", encoding="utf-8")
    assert load_profile(Profile, good).chunking.parent_chars == 2000
    for body, needle in (
        ("chunking: {mode: structure, parent_chars: 2000, extra: 1}\n", "chunking.extra"),
        ("chunking: {mode: structure}\n", "chunking.parent_chars"),
        ("chunking: {mode: structure, parent_chars: 0}\n", "chunking.parent_chars"),
        ("other: 1\nchunking: {mode: structure, parent_chars: 2000}\n", "other"),
    ):
        good.write_text(body, encoding="utf-8")
        with pytest.raises(SettingsError, match=needle) as caught:
            load_profile(Profile, good)
        assert str(good) in str(caught.value)


def test_a_missing_profile_file_is_named(tmp_path):
    with pytest.raises(SettingsError, match="profile .*nope.yaml: cannot be read"):
        load_profile(Profile, tmp_path / "nope.yaml")


@pytest.mark.parametrize("source", ["overlay", "environment"])
def test_a_profile_key_in_the_overlay_or_the_environment_is_refused(tmp_path, source):
    overlay, environ = None, {}
    if source == "overlay":
        overlay = tmp_path / "o.yaml"
        overlay.write_text("chunking: {parent_chars: 1}\n", encoding="utf-8")
    else:
        environ = {"RAG__CHUNKING__PARENT_CHARS": "1"}
    with pytest.raises(SettingsError, match=r"chunking\.parent_chars: a profile \(P\) key"):
        load(Overlay, overlay=overlay, environ=environ, secrets_dir=tmp_path, profile_sections={"chunking"})


def test_an_open_section_takes_any_key_and_a_closed_one_does_not(tmp_path):
    from pydantic import ConfigDict

    class Open(OverlaySection):
        model_config = ConfigDict(extra="allow", frozen=True)

    class Both(OverlaySection):
        open: Open = Open()
        net: Net = Net()

    overlay = tmp_path / "o.yaml"
    overlay.write_text("open: {anything: {url: 'http://y'}}\n", encoding="utf-8")
    assert load(Both, overlay=overlay, environ={}, secrets_dir=tmp_path).open.model_extra == {
        "anything": {"url": "http://y"}
    }
    overlay.write_text("net: {anything: 1}\n", encoding="utf-8")
    with pytest.raises(SettingsError, match=r"net\.anything: unknown key"):
        load(Both, overlay=overlay, environ={}, secrets_dir=tmp_path)
