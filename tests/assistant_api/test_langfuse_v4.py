"""The backend's Langfuse SDK v4 wiring, read at source level (the handler and run metadata live in graph/tracing.py)."""

import ast
import os

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def tree(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        return ast.parse(fh.read())


def test_the_handler_comes_from_langfuse_langchain():
    # langfuse.callback is gone in v4; the LangChain handler lives in langfuse.langchain
    imports = {
        (node.module, alias.name)
        for node in ast.walk(tree("apps/assistant_api/src/assistant_api/graph/tracing.py"))
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert ("langfuse.langchain", "CallbackHandler") in imports
    assert [module for module, _ in imports if module == "langfuse.callback"] == []


def test_langfuse_tags_is_a_list_so_v4_keeps_the_tag():
    # v4 reads langfuse_tags only as a list (CallbackHandler.py:510-520) and drops a string
    values = [
        value
        for node in ast.walk(tree("apps/assistant_api/src/assistant_api/graph/tracing.py"))
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant) and key.value == "langfuse_tags"
    ]
    assert len(values) == 1 and isinstance(values[0], ast.List) and len(values[0].elts) == 1
