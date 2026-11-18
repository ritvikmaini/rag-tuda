"""The indexer's wiring on generations: the target
comes from the profile, the embeddings wrapper calls the client, the writer's FTS language is
the profile's, and no rename swap or cascade is left. The build command runs in test_cli_build.py, the flow in test_write_and_publish.py,
the SQL in tests/pgvector/test_generations_postgres.py."""

import inspect

from indexer import build, cli
from indexer.index import generations as gen
from indexer.index import writer
from indexer.settings import wire


def test_the_target_comes_from_the_profile(settings):
    wiring = wire(settings, connect=lambda application_name: None)
    assert wiring.target == gen.Target("rag")


def test_no_rename_swap_and_no_cascade_is_left_in_the_indexer():
    for module in (cli, build, writer):
        source = inspect.getsource(module)
        assert "_atomic_table_swap" not in source and "RENAME TO" not in source and "CASCADE" not in source
