#!/bin/bash
# Pipeline completo de preprocesamiento de datasets Catalan
# Paso 1: Extraer audios de Arrow a WAV
# Paso 2: Preprocesar WAV files para IndexTTS

set -e

# Activar entorno virtual
source .venv/bin/activate

echo "========================================"
echo "Pipeline de Preprocesamiento Catalan"
echo "========================================"
echo ""

# Dataset paths
LAFRESCAT_PATH="/home/sergioc/projects/datasets/catalan/lafrescat"
FESTCAT_PATH="/home/sergioc/projects/datasets/catalan/festcat"
OUTPUT_ROOT="processed_catalan"

# ============================================
# PASO 1: Extraer audios de Arrow a WAV
# ============================================

echo "PASO 1: Extrayendo audios de Arrow a WAV..."
echo ""

# Procesar lafrescat
echo "Procesando lafrescat..."
python tools/extract_arrow_to_wav.py \
    --dataset-path "$LAFRESCAT_PATH" \
    --output-dir "$OUTPUT_ROOT/lafrescat_extracted"

echo ""

# Procesar festcat
echo "Procesando festcat..."
python tools/extract_arrow_to_wav.py \
    --dataset-path "$FESTCAT_PATH" \
    --output-dir "$OUTPUT_ROOT/festcat_extracted"

echo ""
echo "========================================"
echo "Extracción completada!"
echo "========================================"
echo ""

# ============================================
# PASO 2: Preprocesar WAV files con IndexTTS
# ============================================

echo "PASO 2: Preprocesando WAV files con IndexTTS..."
echo ""

# Procesar lafrescat
echo "Preprocesando lafrescat..."
python tools/preprocess_data.py \
    --manifest "$OUTPUT_ROOT/lafrescat_extracted/manifest.jsonl" \
    --output-dir "$OUTPUT_ROOT/lafrescat_processed" \
    --tokenizer checkpoints/bpe.model \
    --config checkpoints/config.yaml \
    --gpt-checkpoint checkpoints/gpt.pth \
    --language ca \
    --device cuda:1 \
    --batch-size 8 \
    --workers 4 \
    --val-ratio 0.05 \
    --skip-existing

echo ""

# Procesar festcat
echo "Preprocesando festcat..."
python tools/preprocess_data.py \
    --manifest "$OUTPUT_ROOT/festcat_extracted/manifest.jsonl" \
    --output-dir "$OUTPUT_ROOT/festcat_processed" \
    --tokenizer checkpoints/bpe.model \
    --config checkpoints/config.yaml \
    --gpt-checkpoint checkpoints/gpt.pth \
    --language ca \
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
echo "Resultados:"
echo "  - lafrescat processed: $OUTPUT_ROOT/lafrescat_processed/"
echo "  - festcat processed: $OUTPUT_ROOT/festcat_processed/"
echo ""
echo "Siguiente paso: Generar pares prompt-target"
echo "  python tools/build_gpt_prompt_pairs.py \\"
echo "    --manifest $OUTPUT_ROOT/lafrescat_processed/train_manifest.jsonl \\"
echo "    --output $OUTPUT_ROOT/lafrescat_processed/train_pairs.jsonl"
