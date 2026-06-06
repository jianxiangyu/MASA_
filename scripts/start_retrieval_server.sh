#!/bin/bash
set -euo pipefail

# ============ 配置 ============
SEARCH_DATA_DIR="./data/searchR1"
RETRIEVAL_INDEX="${SEARCH_DATA_DIR}/e5_Flat.index"
RETRIEVAL_CORPUS="${SEARCH_DATA_DIR}/wiki-18.jsonl"
RETRIEVAL_MODEL="./model/e5-base-v2"
RETRIEVAL_SERVER_SCRIPT="./examples/search/retriever/retrieval_server.py"
RETRIEVAL_PORT=8030

# 允许通过命令行覆盖端口
if [[ "${1:-}" == "--port" && -n "${2:-}" ]]; then
    RETRIEVAL_PORT="$2"
fi

# ============ 启动服务 ============
echo "   Port: ${RETRIEVAL_PORT}"
echo "   Index: ${RETRIEVAL_INDEX}"
echo "   Corpus: ${RETRIEVAL_CORPUS}"
echo "   Model: ${RETRIEVAL_MODEL}"

CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 \
    python3 "${RETRIEVAL_SERVER_SCRIPT}" \
    --index_path "${RETRIEVAL_INDEX}" \
    --corpus_path "${RETRIEVAL_CORPUS}" \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model "${RETRIEVAL_MODEL}" \
    --port ${RETRIEVAL_PORT}
