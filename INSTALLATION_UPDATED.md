# Instalación Actualizada - Compatible con GPUs Blackwell

## Cambios Realizados

### ✅ Dependencias Actualizadas en `pyproject.toml`

1. **Flash Attention - Ahora Opcional**
   - ❌ Removido: wheel hardcoded incompatible con Linux/Blackwell
   - ✅ Ahora: opcional via `--extra flashattn`

2. **Versiones Flexibilizadas**:
   - `transformers>=4.52.1` (antes: `==4.52.1`)
   - `accelerate>=1.8.1` (antes: `==1.8.1`)
   - `numpy>=1.26.2,<2.0` (antes: `==1.26.2`)
   - `safetensors>=0.5.2` (antes: `==0.5.2`)
   - `tokenizers>=0.21.0` (antes: `==0.21.0`)

## Instalación

### Opción 1: Instalación Estándar (Sin Flash Attention)

```bash
# Funciona en todas las GPUs, incluyendo Blackwell
uv sync
```

**Notas:**
- ✅ Funciona inmediatamente
- ⚠️ Transformers usarán atención PyTorch nativa (más lento)
- ⚠️ Ver warning al inicio sobre Flash Attention

### Opción 2: Con Flash Attention (Recomendado para GPUs modernas)

#### Para RTX 4090, RTX 3090, A100 (hasta sm_90/Ada):
```bash
# Flash Attention disponible via PyPI
uv sync --extra flashattn
```

#### Para RTX 6000 Blackwell (sm_100):
```bash
# Paso 1: Instalación base
uv sync

# Paso 2: Build Flash Attention desde código fuente
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention

# Build con soporte para Blackwell (sm_100)
MAX_JOBS=4 FLASH_ATTENTION_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST="10.0" python setup.py install

cd ..
```

**Verificar instalación:**
```bash
python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')"
```

### Opción 3: Instalación Completa con Todas las Extras

```bash
# WebUI + DeepSpeed + Flash Attention
uv sync --all-extras
```

**⚠️ Nota para Blackwell:** El extra `flashattn` puede fallar. Si falla:
1. Ignora el error
2. Sigue las instrucciones de "Opción 2 para Blackwell" arriba

## Configuración de GPU (NUEVO)

### Primera Ejecución - Selección Interactiva

Al ejecutar por primera vez, IndexTTS detectará automáticamente todas las GPUs disponibles y te permitirá elegir cuál usar:

```bash
python webui.py
```

**Ejemplo con múltiples GPUs:**
```
🚀 IndexTTS GPU Configuration
==================================================
📍 Platform: Linux (o WSL2 on Windows)

🎮 Detected 2 GPU(s):

  [0] NVIDIA GeForce RTX 4090
      Architecture: Ada Lovelace (sm_8.9)
      Memory: 24.0 GB
      Suggested workers: 3

  [1] NVIDIA RTX 6000 Ada Generation
      Architecture: Blackwell (sm_10.0)
      Memory: 48.0 GB
      Suggested workers: 6
      💎 Blackwell GPU detected!
         • BF16 recommended for stability
         • Flash Attention: build from source required

⚡ Flash Attention: Not installed
   Install with: uv sync --extra flashattn
   ⚠️  Blackwell detected: Build from source required!

🎯 Select GPU to use [0-1]: 1

✅ Configuration saved to: ~/.indextts/gpu_config.json
   Selected GPU: NVIDIA RTX 6000 Ada Generation
```

**Tu selección se guarda y no se volverá a preguntar.**

### Forzar GPU Específica

```bash
# Usar GPU 1 siempre
python webui.py --gpu 1

# O con webui_parallel
python webui_parallel.py --gpu 1
```

### Cambiar GPU Seleccionada

```bash
# Opción 1: Borrar configuración
rm ~/.indextts/gpu_config.json
python webui.py  # Preguntará de nuevo

# Opción 2: Usar argumento --gpu
python webui.py --gpu 0  # Cambiar a GPU 0
```

## Verificación de la Instalación

### 1. Verificar GPU y Dependencias

```bash
python webui.py --verbose
```

Después de la configuración interactiva, verás:
```
📊 IndexTTS Configuration Summary
==================================================
Platform: Linux

GPU 1: NVIDIA RTX 6000 Ada Generation
  • Architecture: Blackwell (sm_10.0)
  • Memory: 48.0 GB
  • CUDA: 12.8
  • PyTorch: 2.8.x

⚡ Flash Attention: v2.x.x  # O "Not installed"

💡 Recommendations:
  • Suggested parallel workers: 6
  • Use BF16 for better stability (FP16 may cause NaN)
  • Build Flash Attention from source for best performance
==================================================
```

### 2. Verificar Transformers

```bash
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

Debería mostrar versión >= 4.52.1

### 3. Verificar torch

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Problemas Comunes

### Error: "flash-attn wheel not found"

**Si ves esto durante `uv sync --extra flashattn`:**

```
No compatible wheel found for flash-attn>=2.8.0
```

**Solución:**
1. Ignora y usa instalación sin Flash Attention: `uv sync`
2. O build desde fuente (ver Opción 2 arriba)

### Error: "CUDA capability not supported"

**Si ves esto con Flash Attention:**

```
RuntimeError: flash_attn does not support compute capability 10.0
```

**Causa:** Wheel pre-compilado no soporta Blackwell

**Solución:** Build desde fuente con `TORCH_CUDA_ARCH_LIST="10.0"`

### Error: "ImportError: cannot import name SeamlessM4TFeatureExtractor"

**Si ves esto:**

```
ImportError: cannot import name 'SeamlessM4TFeatureExtractor' from 'transformers'
```

**Solución:**
```bash
# Actualizar transformers
uv sync --upgrade-package transformers
```

### Warning: "FutureWarning: weights_only=False"

**Esto es normal y seguro.** Se refiere a cargar checkpoints del modelo.

**Si quieres silenciarlo:**
```bash
export PYTHONWARNINGS="ignore::FutureWarning"
python webui.py
```

## Recomendaciones por GPU

### RTX 6000 Blackwell (48GB VRAM)
```bash
# Instalación
uv sync
# Build Flash Attention desde fuente (ver arriba)

# Uso
python webui_parallel.py --verbose
# Usar 4-6 workers en parallel
# Considerar --is_fp16 (pero monitorear estabilidad)
```

### RTX 4090 (24GB VRAM)
```bash
# Instalación
uv sync --extra flashattn

# Uso
python webui_parallel.py --verbose
# Usar 2-3 workers en parallel
# --is_fp16 funciona bien
```

### RTX 3090 (24GB VRAM)
```bash
# Instalación
uv sync --extra flashattn

# Uso
python webui.py --verbose
# Usar 2 workers en parallel si usas webui_parallel.py
# --is_fp16 funciona bien
```

## Actualizar Dependencias

Para actualizar a las últimas versiones compatibles:

```bash
# Actualizar todas las dependencias dentro de los constraints
uv sync --upgrade

# O actualizar paquetes específicos
uv sync --upgrade-package transformers
uv sync --upgrade-package accelerate
```

## Migración desde Versión Anterior

Si ya tenías la versión anterior instalada:

```bash
# 1. Backup tu entorno actual (opcional)
cp -r .venv .venv.backup

# 2. Limpiar lockfile viejo
rm uv.lock

# 3. Reinstalar con nuevas dependencias
uv sync

# 4. Si necesitas Flash Attention para Blackwell
# Sigue las instrucciones de Opción 2 arriba
```

## Verificación Completa

Script de verificación completo:

```python
import torch
import transformers
import gradio as gr

print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"Gradio: {gr.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")

    try:
        import flash_attn
        print(f"Flash Attention: {flash_attn.__version__} ✅")
    except ImportError:
        print("Flash Attention: Not installed ⚠️")
```

Guarda esto como `check_install.py` y ejecuta:
```bash
python check_install.py
```

## Soporte

- Documentación completa: Ver `GPU_TROUBLESHOOTING.md`
- Issues de compatibilidad GPU: Ver sección "Common Issues with Modern GPUs"
- Problemas de OOM: La aplicación ahora maneja automáticamente y sugiere ajustes
