from rag_testing import index_view

COLLECTION = index_view.COLLECTION


def test_the_generation_names_are_the_collection_and_the_number():
    assert index_view.generation_table(1) == f"{COLLECTION}_g0001"
    assert index_view.generation_table(12) == f"{COLLECTION}_g0012"
    assert index_view.parents_of(f"{COLLECTION}_g0012") == f"{COLLECTION}_g0012_sections"
    assert index_view.children_of(f"{COLLECTION}_g0012") == f"{COLLECTION}_g0012_chunks"
    assert index_view.GENERATION_NAME.fullmatch(f"{COLLECTION}_g0007")
