#!/bin/bash
# test_inference.sh - Download a small GGUF model and test hesa-llm inference
#
# Engine requirements (from source code analysis):
#   GGUF v3 with llama.cpp tensor naming:
#   - token_embd.weight (embedding table, must be F32)
#   - output.weight or lm_head.weight (LM head, F32 preferred, or tied to token_embd)
#   - output_norm.weight or norm.weight (final RMSNorm)
#   - blk.{n}.attn_norm.weight   (per-layer RMSNorm before attention)
#   - blk.{n}.attn_q.weight      (query projection)
#   - blk.{n}.attn_k.weight      (key projection)
#   - blk.{n}.attn_v.weight      (value projection)
#   - blk.{n}.attn_output.weight (attention output projection)
#   - blk.{n}.ffn_norm.weight    (per-layer RMSNorm before FFN)
#   - blk.{n}.ffn_gate.weight    (SwiGLU gate)
#   - blk.{n}.ffn_up.weight      (SwiGLU up projection)
#   - blk.{n}.ffn_down.weight    (SwiGLU down projection)
#   Supports: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K-Q6_K
#   Metadata: general.architecture, general.vocab_size, general.block_count,
#             general.context_length, {arch}.embedding_length, etc.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/models"
HEXA_BIN="${SCRIPT_DIR}/build/hesa-llm"
TOKENIZER_PATH="${SCRIPT_DIR}/tokenizer.model"

mkdir -p "$MODEL_DIR"

echo "========================================"
echo " hesa-llm Inference Test"
echo "========================================"
echo ""

# ---- Check binary ----
if [ ! -x "$HEXA_BIN" ]; then
    echo "ERROR: hesa-llm binary not found at $HEXA_BIN"
    echo "Run: cd ~/hesa-llm && cmake -B build && cmake --build build"
    exit 1
fi
echo "[BINARY OK] $HEXA_BIN"

# ---- Model download ----
MODEL_FILE=""
MODEL_NAME=""

# Check for existing models
for f in ${MODEL_DIR}/*.gguf; do
    [ -f "$f" ] || continue
    MODEL_FILE="$(basename "$f")"
    MODEL_NAME="$MODEL_FILE"
    echo "[FOUND] Existing model: $MODEL_FILE"
    break
done

# Download if needed
if [ -z "$MODEL_FILE" ]; then
    echo ""
    echo "Downloading a small GGUF model..."
    echo ""

    # Try SmolLM2-135M first (smallest useful, ~85MB Q4)
    echo "  Trying SmolLM2-135M-Instruct Q4_K_M..."
    SMOL_URL="https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf"
    MODEL_FILE="smollm2-135m-instruct-q4_k_m.gguf"
    HTTP_CODE=$(curl -fL --head -s -o /dev/null -w "%{http_code}" "$SMOL_URL" 2>/dev/null || echo "000")
    echo "    HTTP: $HTTP_CODE"

    if echo "$HTTP_CODE" | grep -q "^[23]"; then
        curl -fL --progress-bar -o "${MODEL_DIR}/${MODEL_FILE}" "$SMOL_URL"
        MODEL_NAME="SmolLM2-135M-Instruct (Q4_K_M)"
        echo "  Done: $MODEL_FILE"
    fi

    # Fallback: Qwen2.5-0.5B-Instruct Q4_K_M (~350MB)
    if [ ! -f "${MODEL_DIR}/${MODEL_FILE}" ]; then
        echo ""
        echo "  Trying Qwen2.5-0.5B-Instruct Q4_K_M..."
        QWEN_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
        MODEL_FILE="qwen2.5-0.5b-instruct-q4_k_m.gguf"
        HTTP_CODE=$(curl -fL --head -s -o /dev/null -w "%{http_code}" "$QWEN_URL" 2>/dev/null || echo "000")
        echo "    HTTP: $HTTP_CODE"

        if echo "$HTTP_CODE" | grep -q "^[23]"; then
            curl -fL --progress-bar -o "${MODEL_DIR}/${MODEL_FILE}" "$QWEN_URL"
            MODEL_NAME="Qwen2.5-0.5B-Instruct (Q4_K_M)"
            echo "  Done: $MODEL_FILE"
        fi
    fi

    # Fallback: TinyLlama-1.1B Chat Q4_K_M (~638MB)
    if [ ! -f "${MODEL_DIR}/${MODEL_FILE}" ]; then
        echo ""
        echo "  Trying TinyLlama-1.1B-Chat-v1.0 Q4_K_M..."
        TINY_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        MODEL_FILE="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        HTTP_CODE=$(curl -fL --head -s -o /dev/null -w "%{http_code}" "$TINY_URL" 2>/dev/null || echo "000")
        echo "    HTTP: $HTTP_CODE"

        if echo "$HTTP_CODE" | grep -q "^[23]"; then
            curl -fL --progress-bar -o "${MODEL_DIR}/${MODEL_FILE}" "$TINY_URL"
            MODEL_NAME="TinyLlama-1.1B-Chat-v1.0 (Q4_K_M)"
            echo "  Done: $MODEL_FILE"
        fi
    fi

    if [ ! -f "${MODEL_DIR}/${MODEL_FILE}" ]; then
        echo ""
        echo "ERROR: Could not download any model automatically."
        echo ""
        echo "Download a GGUF model manually and place it in:"
        echo "  ${MODEL_DIR}/"
        echo ""
        echo "The engine expects llama.cpp GGUF tensor naming (blk.{n}.attn_*, etc.)"
        echo "Try: bartowski/SmolLM2-135M-Instruct-GGUF on huggingface.co"
        exit 1
    fi
fi

MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"
MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
echo ""
echo "========================================"
echo " Model: ${MODEL_NAME}"
echo " File:  ${MODEL_FILE} (${MODEL_SIZE})"
echo "========================================"
echo ""

# ---- Run inference ----
PROMPT="Hello, world!"
MAX_TOKENS=32
TEMP=0.7

echo "Configuration:"
echo "  Prompt:      \"${PROMPT}\""
echo "  Max tokens:  ${MAX_TOKENS}"
echo "  Temperature: ${TEMP}"
echo "  Seed:        42"
echo ""
echo "--- Engine Output ---"
echo ""

CMD="$HEXA_BIN -m $MODEL_PATH -p \"$PROMPT\" -n $MAX_TOKENS -t $TEMP -s 42"
echo "Running: $CMD"
echo ""

# Run inference, capture stdout and stderr separately
OUTPUT=$($CMD 2>&1) || EXIT_CODE=$?
EXIT_CODE=${EXIT_CODE:-0}

echo "$OUTPUT"
echo ""

# ---- Results ----
echo "========================================"
echo " Results"
echo "========================================"
echo "  Exit code:   ${EXIT_CODE}"
echo "  Model:       ${MODEL_NAME} (${MODEL_SIZE})"
echo "  Prompt:      \"${PROMPT}\""
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "  STATUS: SUCCESS"
else
    echo "  STATUS: FAILED (exit code ${EXIT_CODE})"
fi

echo ""
echo "========================================"
echo " Test complete. Full log at: /tmp/hesa_test_$(date +%Y%m%d_%H%M%S).log"
echo "========================================"

# Write summary
cat > /tmp/model_test_summary.txt << EOF
=== hesa-llm Inference Test Summary ===
Date: $(date)
Binary: $HEXA_BIN
Model: $MODEL_NAME ($MODEL_FILE, $MODEL_SIZE)
Architecture requirements (from source analysis):
  GGUF v3, llama.cpp tensor naming convention
  Required tensors:
    - token_embd.weight (embedding, F32 required by engine.cpp:161)
    - output.weight OR lm_head.weight (LM head)
    - output_norm.weight OR norm.weight (final RMSNorm)
    - blk.{n}.attn_norm.weight, blk.{n}.attn_q/k/v.weight, blk.{n}.attn_output.weight
    - blk.{n}.ffn_norm.weight, blk.{n}.ffn_gate.weight, blk.{n}.ffn_up.weight, blk.{n}.ffn_down.weight
  Supported dtypes: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K-Q6_K
  Tokenizer: SentencePiece (tokenizer.model) or GGUF embedded vocab

Test Result:
  Exit code: ${EXIT_CODE}
  Command: $CMD
  Full output:
$OUTPUT
EOF

echo ""
echo "Findings written to /tmp/model_test_summary.txt"
