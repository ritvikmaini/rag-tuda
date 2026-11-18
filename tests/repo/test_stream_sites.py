"""Only the frame encoder writes 'data:': every frame of the assistant is a model."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_no_module_but_the_encoder_writes_a_frame_string():
    app = ROOT / "apps" / "assistant_api" / "src" / "assistant_api"
    writers = [p.relative_to(ROOT).as_posix() for p in app.rglob("*.py") if '"data: "' in p.read_text(encoding="utf-8")]
    assert writers == ["apps/assistant_api/src/assistant_api/sse/frames.py"]
