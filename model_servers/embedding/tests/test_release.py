"""After a large batch (an index build) PyTorch's cached GPU blocks go back to the card; the 1-6 query texts of an answer
never trigger it."""

from embedding_server.main import release_cached_memory


def test_cached_memory_is_released_only_after_large_cuda_batches():
    calls = []
    assert release_cached_memory(512, "cuda", lambda: calls.append(1), 64) is True and calls == [1]
    assert release_cached_memory(6, "cuda", lambda: calls.append(2), 64) is False  # an answer's queries
    assert release_cached_memory(512, "cpu", lambda: calls.append(3), 64) is False  # no GPU
    assert release_cached_memory(512, "cuda", lambda: 1 / 0, 64) is False  # a failing release never fails the batch
    assert calls == [1]
