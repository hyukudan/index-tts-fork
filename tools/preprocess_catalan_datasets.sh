#!/bin/bash
# Script helper para preprocesar los datasets de Catalan (lafrescat y festcat)

set -e

# Rutas de datasets
LAFRESCAT_PATH="/home/sergioc/projects/datasets/catalan/lafrescat"
FESTCAT_PATH="/home/sergioc/projects/datasets/catalan/festcat"

# Directorio de salida
OUTPUT_ROOT="processed_catalan"

# Tokenizer y checkpoints
TOKENIZER="checkpoints/bpe.model"
CONFIG="checkpoints/config.yaml"
GPT_CHECKPOINT="checkpoints/gpt.pth"

# Activar entorno virtual
source .venv/bin/activate

echo "========================================"
echo "Preprocesando datasets de Catalan"
echo "========================================"
echo ""
echo "Datasets:"
echo "  - lafrescat: $LAFRESCAT_PATH"
echo "  - festcat: $FESTCAT_PATH"
echo ""
echo "Output: $OUTPUT_ROOT"
echo "========================================"
echo ""

# Opción 1: Procesar ambos datasets a la vez (recomendado)
echo "Procesando ambos datasets con Blackwell (GPU 1)..."
python tools/preprocess_catalan.py \
    --dataset ca="$LAFRESCAT_PATH" \
    --dataset ca="$FESTCAT_PATH" \
    --output-root "$OUTPUT_ROOT" \
    --tokenizer "$TOKENIZER" \
    --config "$CONFIG" \
    --gpt-checkpoint "$GPT_CHECKPOINT" \
    --device cuda:1 \
    --batch-size 8 \
    --workers 4 \
    --val-ratio 0.05 \
    --skip-existing

echo ""
echo "========================================"
echo "Preprocesamiento completado!"
echo "========================================"
echo ""
echo "Manifests generados:"
echo "  - $OUTPUT_ROOT/lafrescat/train_manifest.jsonl"
echo "  - $OUTPUT_ROOT/lafrescat/val_manifest.jsonl"
echo "  - $OUTPUT_ROOT/festcat/train_manifest.jsonl"
echo "  - $OUTPUT_ROOT/festcat/val_manifest.jsonl"
echo ""
echo "Siguiente paso: Generar pares prompt-target para training GPT"
echo "  python tools/build_gpt_prompt_pairs.py --manifest $OUTPUT_ROOT/lafrescat/train_manifest.jsonl --output $OUTPUT_ROOT/lafrescat/train_pairs.jsonl"
echo "  python tools/build_gpt_prompt_pairs.py --manifest $OUTPUT_ROOT/festcat/train_manifest.jsonl --output $OUTPUT_ROOT/festcat/train_pairs.jsonl"
