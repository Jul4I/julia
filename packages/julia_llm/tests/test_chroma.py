import pytest
from julia_llm.vector_store.chroma import get_top_k_chroma_embeddings


# Test for get_top_k_chroma_embeddings
def test_get_top_k_chroma_embeddings(mocker) -> None:
    # Patch TextLoader.load to return a dummy list of documents.
    dummy_text = [{"page_content": "dummy text"}]
    mocker.patch("julia_llm.vector_store.chroma.TextLoader.load", return_value=dummy_text)

    # Patch the text splitter to return dummy splits.
    dummy_splits = [{"page_content": "dummy split"}]
    mocker.patch(
        "julia_llm.vector_store.chroma.RecursiveCharacterTextSplitter.split_documents",
        return_value=dummy_splits,
    )

    # Create dummy retriever that returns a known result.
    class DummyRetriever:
        def get_relevant_documents(self, query: str):
            # Return a list with one dummy Document-like dictionary.
            return [{"page_content": "result dummy"}]

    # Create a dummy Chroma class with an as_retriever method.
    class DummyChroma:
        def as_retriever(self, search_kwargs: dict):
            # You can assert that k is passed correctly if necessary.
            return DummyRetriever()

    # Patch Chroma.from_documents to return an instance of DummyChroma.
    mocker.patch(
        "julia_llm.vector_store.chroma.Chroma.from_documents",
        return_value=DummyChroma(),
    )

    query = "dummy query"
    path = "dummy path"
    k = 1
    result = get_top_k_chroma_embeddings(query, path, k)

    expected_result = [{"page_content": "result dummy"}]
    assert result == expected_result, f"Expected {expected_result} but got {result}"


if __name__ == "__main__":
    pytest.main()
