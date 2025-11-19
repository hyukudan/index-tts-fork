# Guía de Entrenamiento y Fine-tuning - IndexTTS-2

Esta guía explica cómo entrenar o hacer fine-tuning de modelos IndexTTS-2 para nuevos idiomas **sin sobrescribir los modelos base**.

## 🎯 Objetivo: Preservar Modelos Base

Cuando entrenas para un nuevo idioma (ej: catalán, francés, etc.), **NUNCA** sobrescribirás los modelos originales en `checkpoints/`. En su lugar:

1. Entrenarás en un directorio separado
2. Copiarás el mejor modelo con un nombre descriptivo
3. El sistema de auto-detección lo encontrará automáticamente

## 📂 Estructura de Directorios Recomendada

```
index-tts-fork/
├── checkpoints/                          # Modelos base (NO MODIFICAR)
│   ├── gpt.pth                          # ← Modelo base original
│   ├── bpe.model                        # ← Tokenizer base original
│   └── config.yaml
│
├── training/                             # Directorio de entrenamiento
│   ├── catalan/                         # ← Entrenamiento catalán
│   │   ├── tokenizer/
│   │   │   ├── catalan_bpe.model        # Tokenizer catalán
│   │   │   └── catalan_bpe.vocab
│   │   ├── checkpoints/
│   │   │   ├── model_step1000.pth       # Checkpoints intermedios
│   │   │   ├── model_step2000.pth
│   │   │   └── latest.pth               # Último checkpoint
│   │   └── logs/                        # TensorBoard logs
│   │
│   ├── french/                          # ← Entrenamiento francés
│   │   ├── tokenizer/
│   │   │   ├── french_bpe.model
│   │   │   └── french_bpe.vocab
│   │   └── checkpoints/
│   │       └── ...
│   │
│   └── multilingual/                    # ← Multilingual
│       └── ...
│
└── models/                               # Modelos finalizados (para uso)
    ├── gpt_catalan.pth                  # ← Copia del mejor checkpoint
    ├── catalan_bpe.model                # ← Tokenizer catalán
    ├── gpt_french.pth
    ├── french_bpe.model
    └── ...
```

## 🔧 Paso 1: Entrenar Tokenizer BPE

Para un nuevo idioma, primero crea un tokenizer específico:

```bash
# Ejemplo: Tokenizer para catalán
python tools/tokenizer/train_bpe.py \
    --manifest data/catalan_dataset/train.jsonl \
    --output-prefix training/catalan/tokenizer/catalan_bpe \
    --vocab-size 12000 \
    --character-coverage 0.9995
```

**Resultado:**
- `training/catalan/tokenizer/catalan_bpe.model` ✓
- `training/catalan/tokenizer/catalan_bpe.vocab` ✓

**✅ El modelo base `checkpoints/bpe.model` NO se toca**

## 🚀 Paso 2: Fine-tune del Modelo GPT

Entrena el modelo usando el tokenizer específico:

```bash
# Ejemplo: Fine-tuning para catalán
python trainers/train_gpt_v2.py \
    --train-manifest data/catalan_dataset/train_paired.jsonl \
    --val-manifest data/catalan_dataset/val_paired.jsonl \
    --tokenizer training/catalan/tokenizer/catalan_bpe.model \
    --config checkpoints/config.yaml \
    --base-checkpoint checkpoints/gpt.pth \
    --output-dir training/catalan/checkpoints \
    --batch-size 4 \
    --epochs 10 \
    --learning-rate 2e-5
```

**Argumentos clave:**
- `--tokenizer`: Tu tokenizer específico (NO el base)
- `--base-checkpoint`: Modelo base para fine-tuning
- `--output-dir`: Directorio separado (NO `checkpoints/`)

**Resultado:**
- Modelos en `training/catalan/checkpoints/model_step*.pth`
- Logs en `training/catalan/checkpoints/runs/`

**✅ El modelo base `checkpoints/gpt.pth` NO se modifica**

## 📦 Paso 3: Finalizar y Publicar Modelo

Una vez tengas el mejor checkpoint (ej: `model_step5000.pth`), usa el script helper:

```bash
# Instalar modelo finalizado para uso en WebUI
python tools/install_trained_model.py \
    --checkpoint training/catalan/checkpoints/model_step5000.pth \
    --tokenizer training/catalan/tokenizer/catalan_bpe.model \
    --output-name catalan \
    --description "Catalan fine-tuned model"
```

Este script:
1. Copia el checkpoint a `models/gpt_catalan.pth`
2. Copia el tokenizer a `models/catalan_bpe.model`
3. Actualiza el registro de modelos

**Alternativamente (manual):**

```bash
# Crear directorio de modelos
mkdir -p models

# Copiar checkpoint con nombre descriptivo
cp training/catalan/checkpoints/model_step5000.pth models/gpt_catalan.pth

# Copiar tokenizer con nombre específico
cp training/catalan/tokenizer/catalan_bpe.model models/catalan_bpe.model
```

## 🎨 Convención de Nombres

Para que el sistema de auto-detección funcione correctamente:

### ✅ Opción 1: Mismo nombre base
```
models/
├── gpt_catalan.pth
└── gpt_catalan_bpe.model  ← Auto-detectado por coincidencia de nombre
```

### ✅ Opción 2: Nombre estándar en mismo directorio
```
models/
├── gpt_catalan.pth
└── catalan_bpe.model     ← También funciona si contiene "catalan"
```

### ✅ Opción 3: Subdirectorio tokenizers
```
models/
├── gpt_catalan.pth
└── tokenizers/
    └── catalan_bpe.model  ← Auto-detectado en subdirectorio
```

### ❌ NO HAGAS ESTO:
```
checkpoints/
├── gpt.pth              ← SOBRESCRITO (MAL!)
└── bpe.model            ← PERDIDO (MAL!)
```

## 🖥️ Uso en WebUI

Después de instalar el modelo, aparecerá automáticamente en la WebUI:

1. **Dropdown "Model Checkpoint":**
   ```
   gpt.pth (3.2GB, v2.0, zh/en)           ← Original
   gpt_catalan.pth (3.2GB, v2.0, ca)      ← Tu modelo
   gpt_french.pth (3.2GB, v2.0, fr)       ← Otro modelo
   ```

2. **Metadata auto-detectada:**
   ```
   Tokenizer: catalan_bpe.model (12000 vocab)
   VRAM: 8.1 GB
   ```

3. **GPU selector:** Elige GPU 0 o GPU 1

4. **Load Model:** Carga con hot-swap

## 🔄 Comparar Modelos

En la pestaña "Compare Models":

```
Model A: gpt.pth (original)
Model B: gpt_catalan.pth (catalán)

→ Genera con ambos
→ Compara RTF, calidad, métricas
```

## 📋 Ejemplo Completo: Añadir Catalán

```bash
# 1. Preparar datos (ver tools/prepare_dataset.py)
python tools/prepare_dataset.py \
    --audio-dir data/raw/catalan_audio/ \
    --transcript-file data/raw/catalan_transcripts.txt \
    --output-manifest data/catalan_dataset/train.jsonl

# 2. Crear pares prompt/target para GPT
python tools/build_gpt_prompt_pairs.py \
    --manifest data/catalan_dataset/train.jsonl \
    --output data/catalan_dataset/train_paired.jsonl

# 3. Entrenar tokenizer
python tools/tokenizer/train_bpe.py \
    --manifest data/catalan_dataset/train.jsonl \
    --output-prefix training/catalan/tokenizer/catalan_bpe \
    --vocab-size 12000

# 4. Fine-tune modelo
python trainers/train_gpt_v2.py \
    --train-manifest data/catalan_dataset/train_paired.jsonl \
    --val-manifest data/catalan_dataset/val_paired.jsonl \
    --tokenizer training/catalan/tokenizer/catalan_bpe.model \
    --base-checkpoint checkpoints/gpt.pth \
    --output-dir training/catalan/checkpoints \
    --epochs 10

# 5. Instalar mejor checkpoint (ej: step 5000)
python tools/install_trained_model.py \
    --checkpoint training/catalan/checkpoints/model_step5000.pth \
    --tokenizer training/catalan/tokenizer/catalan_bpe.model \
    --output-name catalan

# 6. Usar en WebUI
python webui.py --model-dir models
```

## ⚠️ Checklist de Seguridad

Antes de entrenar, verifica:

- [ ] `--output-dir` NO es `checkpoints/`
- [ ] `--tokenizer` apunta a TU tokenizer, no al base
- [ ] Tienes backup de `checkpoints/` original
- [ ] El nombre del modelo final incluye el idioma (ej: `gpt_catalan.pth`)
- [ ] El tokenizer final tiene nombre relacionado (ej: `catalan_bpe.model`)

## 🎓 Tips Avanzados

### Múltiples idiomas en un modelo

```bash
# Combinar datasets
python trainers/train_gpt_v2.py \
    --train-manifest data/catalan/train.jsonl::ca \
    --train-manifest data/spanish/train.jsonl::es \
    --train-manifest data/french/train.jsonl::fr \
    --output-dir training/multilingual_cat_es_fr \
    --output-name multilingual_romance
```

### Continuar entrenamiento

```bash
python trainers/train_gpt_v2.py \
    --resume training/catalan/checkpoints/latest.pth \
    --epochs 20  # Continuar más épocas
```

### Usar GPU específica

```bash
CUDA_VISIBLE_DEVICES=1 python trainers/train_gpt_v2.py \
    --output-dir training/catalan/checkpoints \
    ...
```

## 📚 Recursos Adicionales

- **Preparación de datos:** `tools/README.md`
- **Tokenizer custom:** `tools/tokenizer/README.md`
- **Training avanzado:** `trainers/README.md`
- **WebUI features:** `docs/WEBUI_GUIDE.md`

---

**Regla de Oro:** Nunca modifiques archivos en `checkpoints/` directamente. Usa directorios separados y copia solo cuando estés seguro del resultado final.
