# INDEX_PATH — BM25 index dir. Override via env; defaults to the in-repo corpus
# clone (see README for cloning BrowseComp-Plus into data/).
INDEX_PATH="${INDEX_PATH:-data/BrowseComp-Plus/indexes/bm25}"

python -m src.run  \
   --mode eval   \
   --benchmark browsecomp-plus  \
   --client litellm   \
   --model gpt-5.4-mini   \
   --index_path "$INDEX_PATH" \
   --run_dir continue_normal_wo_memory_tool \
   --run_id extra_15 \
   --limit 100