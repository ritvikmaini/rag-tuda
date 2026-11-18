"""deduplicate: the first document of each distinct text, whether a LangChain Document, a duck-typed one or a dict."""

from langchain_core.documents import Document

from rag_core.domain.retrieval import deduplicate


class Plain:
    def __init__(self, page_content):
        self.page_content = page_content


def test_the_first_of_each_text_stays_in_order_and_empty_texts_drop_out():
    first, same, other = Document(page_content="a"), Plain("a"), {"text": "b"}
    docs = [first, same, other, {"text": "b"}, Document(page_content=""), {"text": None}, {"metadata": {}}, object()]
    assert deduplicate(docs) == [first, other]
