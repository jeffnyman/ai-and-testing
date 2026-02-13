# Reference:
# ???

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase

loader = PyPDFLoader("./arXiv-jnyman-051011v3.pdf")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=1000,
  chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(chunks, embeddings)

# Set up retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

execution_model = ChatOllama(model="jeffnyman/ts-reasoner")
judge_model = OllamaModel(model="jeffnyman/ts-evaluator")

metric = ContextualPrecisionMetric(model=judge_model, verbose_mode=True)

question = """Please consider Jeff Nyman's warp drive paper.
What energy source does the paper propose would be needed to
generate the warp bubble for faster-than-light travel?"""

# The ground truth / expected answer
expected_output = """Matter/antimatter annihilation, requiring
approximately 10^28 kg of antimatter (equivalent to Jupiter's
mass-energy)."""

# Get relevant context
retrieved_docs = retriever.invoke(question)

print("=" * 60)
print("RETRIEVED CHUNKS")
print("=" * 60)

for i, doc in enumerate(retrieved_docs, 1):
  print(f"\n--- Chunk {i} (Page {doc.metadata.get('page', 'unknown')}) ---")
  print(doc.page_content)
  print()

context = [doc.page_content for doc in retrieved_docs]

prompt = f"Based on this context: {context}\n\nQuestion: {question}"
response = execution_model.invoke(prompt).content

print("=" * 60)
print("GENERATED RESPONSE")
print("=" * 60)
print(response)
print()

test_case = LLMTestCase(
  input=question,
  actual_output=response,
  expected_output=expected_output,
  retrieval_context=context
)

print("=" * 60)
print("CONTEXTUAL PRECISION EVALUATION")
print("=" * 60)
metric.measure(test_case)
