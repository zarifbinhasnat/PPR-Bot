import sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from ppr_bot.config import settings
from ppr_bot.indexing.embedder import Embedder
from ppr_bot.indexing.vector_store import NumpyVectorStore
from ppr_bot.indexing.bm25_index import BM25Index
from ppr_bot.retrieval.hybrid_search import hybrid_search
from ppr_bot.retrieval.reranker import Reranker
import json

q = "সরাসরি ক্রয় কী এবং কখন এটি ব্যবহার করা যায়?"
t=time.time(); emb=Embedder(); _=emb.embed_query("warmup"); print(f"embedder load+warm: {time.time()-t:.1f}s")
t=time.time(); rr=Reranker(); _=rr._model; print(f"reranker load: {time.time()-t:.1f}s")

ids=[]; 
with settings.chunks_path.open(encoding="utf-8") as f:
    chunks=[json.loads(l) for l in f if l.strip()]
ids=[c["chunk_id"] for c in chunks]
vs=NumpyVectorStore.load(settings.embeddings_path, ids)
bm=BM25Index.load(settings.bm25_index_path)

t=time.time(); fused=hybrid_search(q, emb, vs, bm, settings.TOP_K_DENSE, settings.TOP_K_SPARSE, settings.RRF_K); print(f"hybrid_search: {time.time()-t:.1f}s -> {len(fused)} candidates")
by={c['chunk_id']:c for c in chunks}
cand=[by[cid] for cid,_ in fused if cid in by]
t=time.time(); top=rr.rerank(q, cand, top_k=5); print(f"rerank {len(cand)} pairs: {time.time()-t:.1f}s")
print("top result:", top[0]['metadata'].get('breadcrumb','')[:60], "score", round(top[0]['rerank_score'],3))
