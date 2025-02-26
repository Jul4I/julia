from julia_llm.vector_store.chroma import (
    get_top_k_chroma_embeddings,
    invoke_openai_tplt,
)
from julia_log import logger

log = logger.get_logger(__name__)


if __name__ == "__main__":
    query = "What is X?"
    context = get_top_k_chroma_embeddings(
        query=query,
        path="./docs/text.txt",
    )
    result = invoke_openai_tplt(context, query)
    log.info(result)
