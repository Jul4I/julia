from dotenv import load_dotenv
from julia_log.logger import get_logger
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import TextLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

log = get_logger(__name__)

load_dotenv()

embedding_model = "all-MiniLM-L6-v2"
HuggingFaceEmbeddings_ = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False},
)

store = {}


def get_top_k_chroma_embeddings(
    query: str,
    path: str,
    k: int = 5,
) -> list[Document]:
    text_loader = TextLoader(file_path=path).load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    splits = splitter.split_documents(text_loader)

    chroma_vector_database = Chroma.from_documents(
        collection_name="example_collection",
        documents=splits,
        embedding=HuggingFaceEmbeddings_,
        persist_directory="./chroma_langchain_db",
    )
    retriever = chroma_vector_database.as_retriever(search_kwargs={"k": k})

    return retriever.get_relevant_documents(query)


def invoke_openai_tplt(context: list[Document], query: str):  # noqa: ANN201
    template_ = """
        Answer the question based only on the following context: {context}
        Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template_, verbose=True)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    chain = prompt | llm

    return chain.invoke({"context": context, "question": query}).content


def invoke_openai_msg(context: list[Document], query: str):  # noqa: ANN201
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "System",
                "Eres un asistente virtual que ayuda a los usuarios a encontrar información en un documento.",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("Human", "{input}"),
        ],
    )
    llm = ChatOpenAI()
    runnable_chain = prompt | llm

    with_message_history = RunnableWithMessageHistory(  # noqa: F841
        runnable=runnable_chain,  # type: ignore  # noqa: PGH003
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    return runnable_chain.invoke({"context": context, "question": query}).content


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
