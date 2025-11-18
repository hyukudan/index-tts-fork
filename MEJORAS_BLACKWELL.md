# Resumen de Mejoras para GPUs Modernas (Blackwell, Ada, etc.)

## 🎯 Objetivo

Hacer que IndexTTS funcione perfectamente en GPUs modernas como **RTX 6000 Blackwell**, resolviendo problemas con Wav2Vec2Bert, Whisper, Flash Attention, y añadiendo soporte multi-GPU.

---

## ✅ Mejoras Implementadas

### 1. 🎮 Sistema Multi-GPU con Selección Interactiva

**Problema anterior:**
- Solo usaba GPU 0 siempre
- No había forma de elegir GPU en sistemas multi-GPU
- No se detectaba la plataforma (WSL vs Windows vs Linux)

**Solución implementada:**

Nuevo módulo: `indextts/utils/gpu_config.py`

**Primera ejecución:**
```bash
python webui.py
```

```
🚀 IndexTTS GPU Configuration
==================================================
📍 Platform: WSL2 on Windows
   💡 WSL2 often provides better performance than Windows native

🎮 Detected 2 GPU(s):

  [0] NVIDIA GeForce RTX 4090
      Architecture: Ada Lovelace (sm_8.9)
      Memory: 24.0 GB
      Suggested workers: 3
      ✨ Ada Lovelace GPU - excellent performance
         • Flash Attention available via pip

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
      See INSTALLATION_UPDATED.md for instructions

🎯 Select GPU to use [0-1]: 1

✅ Configuration saved to: ~/.indextts/gpu_config.json
   Selected GPU: NVIDIA RTX 6000 Ada Generation
```

**Características:**
- ✅ Detecta TODAS las GPUs disponibles
- ✅ Muestra arquitectura, memoria, compute capability
- ✅ Sugerencia automática de workers según VRAM
- ✅ Detecta Blackwell, Ada, Ampere, Turing, Volta
- ✅ Detecta WSL2 vs Windows vs Linux
- ✅ Guarda configuración para próximas ejecuciones
- ✅ Argumento `--gpu N` para override
- ✅ Detección de Flash Attention

**Uso:**
```bash
# Primera vez: selección interactiva
python webui.py

# Forzar GPU específica
python webui.py --gpu 1

# Reconfigurar
rm ~/.indextts/gpu_config.json
python webui.py
```

---

### 2. 🛡️ Manejo de Errores OOM (Out of Memory)

**Problema anterior:**
- Crashes sin mensajes claros
- No se limpiaba memoria CUDA
- Difícil saber qué ajustar

**Solución implementada:**

**En `webui.py` y `webui_parallel.py`:**
- Try-catch específico para RuntimeError OOM
- Limpieza automática de cache CUDA
- Mensajes claros con sugerencias

**Ejemplo de error mejorado:**
```
⚠️ GPU out of memory. Try reducing max_mel_tokens, max_text_tokens_per_sentence, or duration.
   Error: CUDA out of memory. Tried to allocate 2.50 GiB...
```

**Dónde se aplica:**
- `gen_single()` - Generación individual
- `generate_all_batch()` - Generación batch
- `regenerate_batch_entry()` - Regeneración
- `_worker_loop()` - Workers paralelos

**Función de limpieza:**
```python
def cleanup_gpu_memory():
    """Clean up GPU memory to prevent OOM errors."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
```

Se llama automáticamente después de cada batch.

---

### 3. 🔧 Actualización de Dependencias

**Problema anterior:**
- `flash-attn` hardcoded wheel solo para Windows Python 3.10
- Versiones pinned impedían actualizaciones
- No funcionaba en Linux ni Blackwell

**Cambios en `pyproject.toml`:**

```diff
- "accelerate==1.8.1"
+ "accelerate>=1.8.1"  # Permite updates

- "numpy==1.26.2"
+ "numpy>=1.26.2,<2.0"  # Flexibilidad sin romper

- "transformers==4.52.1"
+ "transformers>=4.52.1"  # Fixes CUDA 12.8

- "safetensors==0.5.2"
+ "safetensors>=0.5.2"

- "tokenizers==0.21.0"
+ "tokenizers>=0.21.0"

- flash-attn = { path = "flash_attn-...-win_amd64.whl" }
+ # Movido a [project.optional-dependencies]
```

**Nuevo extra opcional:**
```toml
[project.optional-dependencies]
flashattn = [
  "flash-attn>=2.8.0; sys_platform == 'linux'",
]
```

**Instalación:**
```bash
# Sin Flash Attention (funciona, más lento)
uv sync

# Con Flash Attention (RTX 4090, 3090)
uv sync --extra flashattn

# Blackwell - build desde fuente
uv sync
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
MAX_JOBS=4 FLASH_ATTENTION_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST="10.0" python setup.py install
```

---

### 4. 🚨 Detección y Warnings Específicos

**Nuevos checks al inicio:**

1. **Arquitectura GPU**
   - Blackwell (sm_10.0+): Tip sobre BF16
   - Ada (sm_8.9): Info sobre Flash Attention
   - Ampere/Turing/Volta: Identificación

2. **Flash Attention**
   - Detecta si está instalado
   - Muestra versión
   - Warning si no está (con instrucciones)

3. **Plataforma**
   - Detecta WSL2 vs Windows vs Linux
   - Recomienda WSL2 si estás en Windows nativo

4. **Memoria GPU**
   - Muestra VRAM total
   - Calcula workers sugeridos

5. **Optimizaciones CUDA**
   - Para Blackwell: activa `CUDA_LAUNCH_BLOCKING=0` y `TORCH_CUDNN_V8_API_ENABLED=1`

---

### 5. 📚 Documentación Completa

#### `GPU_TROUBLESHOOTING.md`
- **Problemas comunes** con Wav2Vec2Bert, Whisper, transformers
- **Soluciones** para Flash Attention en Blackwell
- **Problemas FP16 vs BF16** en Blackwell
- **Errores OOM** y cómo resolverlos
- **Recomendaciones** por modelo de GPU
- **Comandos de debugging**
- **Sección multi-GPU** completa

#### `INSTALLATION_UPDATED.md`
- **Guía de instalación** actualizada
- **Opciones de instalación** por GPU
- **Ejemplos** de selección interactiva
- **Troubleshooting** común
- **Scripts de verificación**
- **Recomendaciones** por GPU
- **Migración** desde versión anterior

#### `MEJORAS_BLACKWELL.md` (este archivo)
- Resumen ejecutivo de todas las mejoras
- Ejemplos de uso
- Casos de uso
- Checklist de verificación

---

## 🎯 Casos de Uso Resueltos

### Caso 1: Sistema con Blackwell + Ada

**Situación:**
- PC con RTX 6000 Blackwell (48GB) + RTX 4090 (24GB)
- Quieres usar Blackwell para producción, Ada para desarrollo

**Solución:**
```bash
# Producción en Blackwell
python webui_parallel.py --gpu 0

# Desarrollo en Ada
python webui.py --gpu 1
```

### Caso 2: Errores con Wav2Vec2Bert en Blackwell

**Síntomas:**
- NaN en outputs
- CUDA errors
- Crashes durante inferencia

**Causas identificadas:**
1. FP16 inestable en Blackwell
2. Flash Attention wheel incompatible
3. transformers versión vieja

**Solución:**
1. Actualizar dependencias: `uv sync`
2. Considerar NO usar `--is_fp16` (o usar BF16 si se implementa)
3. Build Flash Attention desde fuente
4. Ver `GPU_TROUBLESHOOTING.md` sección "Wav2Vec2Bert"

### Caso 3: WSL vs Windows Nativo

**Problema:**
- Rendimiento diferente
- Compatibilidad de Flash Attention

**Solución:**
- Sistema detecta automáticamente WSL
- Recomienda WSL2 si estás en Windows
- Flash Attention funciona mejor en WSL2

---

## 📊 Verificación de Mejoras

### Checklist de Verificación

```bash
# 1. Verificar instalación
uv sync

# 2. Primera ejecución - configuración GPU
python webui.py

# Deberías ver:
# ✅ Selección interactiva de GPU
# ✅ Detección de plataforma
# ✅ Info de todas las GPUs
# ✅ Flash Attention status
# ✅ Recomendaciones específicas

# 3. Verificar config guardada
cat ~/.indextts/gpu_config.json

# 4. Verificar argumento --gpu
python webui.py --gpu 0

# 5. Verificar manejo de OOM
# Intenta generar con parámetros muy altos
# Deberías ver mensaje claro de OOM con sugerencias

# 6. Verificar multi-GPU
nvidia-smi  # Ver que usa la GPU correcta
```

### Script de Verificación

```python
# Guarda como check_mejoras.py
import torch
from indextts.utils.gpu_config import GPUConfig

config = GPUConfig()

print("=== Verificación de Mejoras ===\n")

# 1. Detección de plataforma
platform_info = config.detect_platform()
print(f"1. Plataforma: {'WSL2' if platform_info['is_wsl'] else platform_info['system']}")

# 2. GPUs detectadas
gpus = config.get_gpu_info()
print(f"2. GPUs detectadas: {len(gpus)}")
for gpu in gpus:
    print(f"   [{gpu['id']}] {gpu['name']} - {gpu['architecture']}")

# 3. Flash Attention
flash_info = config.check_flash_attention()
print(f"3. Flash Attention: {'Instalado v' + flash_info['version'] if flash_info['installed'] else 'No instalado'}")

# 4. Config guardada
saved_gpu = config.config.get("selected_gpu_id")
print(f"4. GPU guardada: {saved_gpu if saved_gpu is not None else 'Ninguna (primera ejecución)'}")

# 5. CUDA disponible
print(f"5. CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   PyTorch: {torch.__version__}")

print("\n✅ Verificación completa!")
```

Ejecutar:
```bash
python check_mejoras.py
```

---

## 🔄 Flujo de Trabajo Recomendado

### Para desarrollo en sistema multi-GPU:

```bash
# 1. Primera ejecución - configura GPU favorita
python webui.py
# Selecciona tu GPU de desarrollo

# 2. Desarrollo normal
python webui.py
# Usa GPU guardada automáticamente

# 3. Testear en otra GPU
python webui.py --gpu 1
# Override temporal

# 4. Cambiar GPU permanente
rm ~/.indextts/gpu_config.json
python webui.py
# Reconfigurar
```

### Para sistemas con Blackwell:

```bash
# 1. Instalar dependencias base
uv sync

# 2. Build Flash Attention desde fuente
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
MAX_JOBS=4 FLASH_ATTENTION_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST="10.0" python setup.py install
cd ..

# 3. Primera ejecución
python webui.py
# Verás recomendaciones específicas para Blackwell

# 4. Monitorear memoria durante uso
watch -n 1 nvidia-smi

# 5. Si hay OOM, ajustar parámetros según mensajes
# El sistema sugiere qué reducir
```

---

## 📈 Comparación Antes vs Después

| Aspecto | Antes | Después |
|---------|-------|---------|
| **Multi-GPU** | Solo GPU 0 | Selección interactiva |
| **Detección plataforma** | No | WSL/Windows/Linux |
| **Errores OOM** | Crash sin info | Mensaje claro + sugerencias |
| **Flash Attention** | Wheel hardcoded | Opcional + build instructions |
| **Dependencias** | Versiones fijas | Actualizables dentro de constraints |
| **Blackwell** | Problemas de compatibilidad | Detectado con recomendaciones |
| **Configuración** | Manual en código | Interactiva + persistente |
| **Documentación** | Básica | Completa con troubleshooting |

---

## 🚀 Próximos Pasos Recomendados

### Mejoras futuras posibles:

1. **Soporte BF16 explícito**
   - Agregar opción `--use_bf16` para Blackwell
   - Conversión automática de modelos a BF16

2. **Telemetría de GPU**
   - Logging de uso de VRAM durante inferencia
   - Alertas proactivas antes de OOM

3. **Perfiles de configuración**
   - Perfiles predefinidos por GPU
   - "Blackwell optimized", "Ada balanced", etc.

4. **Benchmark automático**
   - Testear rendimiento en primera ejecución
   - Sugerir parámetros óptimos

5. **Multi-GPU paralelo**
   - Distribuir batch entre múltiples GPUs
   - Balanceo de carga automático

---

## 📞 Soporte

### Si encuentras problemas:

1. **Revisa la documentación:**
   - `GPU_TROUBLESHOOTING.md` - Problemas comunes
   - `INSTALLATION_UPDATED.md` - Instalación paso a paso

2. **Ejecuta verificación:**
   ```bash
   python check_mejoras.py
   ```

3. **Check verbose:**
   ```bash
   python webui.py --verbose
   ```

4. **Info de sistema:**
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.__version__, torch.version.cuda)"
   ```

5. **Reset configuración:**
   ```bash
   rm ~/.indextts/gpu_config.json
   python webui.py
   ```

---

## ✨ Créditos

Estas mejoras resuelven problemas específicos identificados en sistemas con:
- RTX 6000 Blackwell
- RTX 4090 Ada Lovelace
- WSL2 en Windows
- Linux con CUDA 12.8
- Sistemas multi-GPU

Todas las mejoras son compatibles hacia atrás con GPUs más antiguas (Ampere, Turing, Volta).

---

**Última actualización:** 2025-01-18
**Versión:** 2.0 con soporte multi-GPU y Blackwell
