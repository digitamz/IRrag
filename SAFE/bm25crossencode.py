from sentence_transformers import CrossEncoder
import bm25s
import pickle
from pyserini.search.lucene import LuceneSearcher

#Will need to obtain one, cannot add it to repo due to size.
INDEX_DIR = "wiki_index"
searcher = LuceneSearcher(INDEX_DIR)
searcher.set_bm25(k1=0.9, b=0.4)
def obtain_docs(query: str, num_searches: int = 1000):
    # Search top `num_searches` documents
    hits = searcher.search(query, k=num_searches)

    # Extract document strings
    docs = [searcher.doc(hit.docid).raw() for hit in hits]

    # Rerank and return top 3
    top_docs = rerank(query, docs, 3)

    return top_docs

def rerank(query, docs, number=3):
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
    pairs = [(query, doc) for doc in docs]
    scores = model.predict(pairs, batch_size=32)
    reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    top_docs_only = [doc for doc, score in reranked[:number]]
    return top_docs_only
