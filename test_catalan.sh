#!/bin/bash
# Script para probar el modelo catalán en la Blackwell (GPU 0)

export CUDA_VISIBLE_DEVICES=1

uv run python inference_script.py \
    --config checkpoints/config_catalan.yaml \
    --speaker processed_catalan/lafrescat_extracted/wavs/sample_001111.wav \
    --text "Bon dia, com estàs? Avui fa un dia molt bonic." \
    --output /tmp/test_catalan_final.wav \
    --device cuda:0 \
    --verbose

echo ""
echo "Audio guardado en: /tmp/test_catalan_final.wav"
