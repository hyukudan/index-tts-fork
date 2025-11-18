# Guía de Instalación y Uso en WSL2

## ✅ ¿Funcionará en WSL?

**SÍ**, IndexTTS funciona perfectamente en WSL2 con las siguientes consideraciones:

### Requisitos

1. **WSL2** (NO WSL1)
   - WSL1 NO soporta GPU
   - Verifica tu versión: `wsl --list --verbose`
   - Actualiza a WSL2: `wsl --set-version Ubuntu 2`

2. **Drivers NVIDIA para WSL**
   - Driver Windows ≥ 560.x
   - NO instalar drivers CUDA en WSL
   - Descargar: https://www.nvidia.com/Download/index.aspx

3. **CUDA Toolkit en WSL**
   - Se instala via PyTorch automáticamente
   - NO necesitas instalar CUDA manualmente

## 🚀 Instalación en WSL2

### Paso 1: Verificar WSL2

```bash
# En PowerShell (Windows)
wsl --list --verbose

# Deberías ver:
# NAME      STATE    VERSION
# Ubuntu    Running  2
```

Si VERSION es 1, actualiza:
```bash
wsl --set-version Ubuntu 2
```

### Paso 2: Verificar GPU

```bash
# Dentro de WSL
nvidia-smi
```

Deberías ver tu GPU. Si no:

**Solución:**
```bash
# En PowerShell (Windows)
# 1. Actualizar drivers NVIDIA
# Descarga desde: https://www.nvidia.com/Download/index.aspx

# 2. Reiniciar WSL
wsl --shutdown

# 3. Iniciar WSL de nuevo
wsl
```

### Paso 3: Clonar e Instalar

```bash
# En WSL
cd ~
git clone <tu-repo>
cd index-tts-fork

# Instalar con uv
uv sync
```

### Paso 4: Primera Ejecución

```bash
python webui.py
```

**Verás:**
```
🚀 IndexTTS GPU Configuration
==================================================
📍 Platform: WSL2
   ✅ WSL2 detected - GPU support available
   ✅ NVIDIA Driver: 560.94
   💡 WSL2 often provides better performance than Windows native

🎮 Detected 1 GPU(s):
  [0] NVIDIA RTX 6000 Blackwell
      Architecture: Blackwell (sm_10.0)
      Memory: 48.0 GB
      Suggested workers: 6
      💎 Blackwell GPU detected!
         • WSL: Ensure latest NVIDIA drivers (560+)

🎯 Select GPU to use [0]: 0
✅ Configuration saved to: ~/.indextts/gpu_config.json
```

## ⚠️ Problemas Comunes en WSL

### 1. "nvidia-smi: command not found"

**Causa:** Drivers WSL no instalados

**Solución:**
```bash
# En PowerShell (Windows)
# 1. Actualizar Windows a última versión
# 2. Descargar e instalar drivers NVIDIA para WSL
# 3. Reiniciar WSL
wsl --shutdown
```

### 2. "CUDA not available in PyTorch"

**Causa:** PyTorch no detecta CUDA

**Verificar:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Solución:**
```bash
# Reinstalar PyTorch con CUDA
uv sync --reinstall-package torch
```

### 3. "GPU out of memory" más frecuente que en Linux

**Causa:** WSL comparte memoria con Windows

**Solución:**
```bash
# Reducir workers en webui_parallel.py
python webui_parallel.py  # Usar sugerencia automática
# O forzar menos workers en la UI
```

Configurar `.wslconfig` en Windows:
```ini
# En C:\Users\<tu-usuario>\.wslconfig
[wsl2]
memory=32GB  # Ajusta según tu RAM
processors=8
```

Reiniciar WSL:
```bash
wsl --shutdown
```

### 4. Multiprocessing errors en webui_parallel

**Síntoma:**
```
RuntimeError: context has already been set
```

**Causa:** WSL tiene peculiaridades con fork/spawn

**Solución:** Ya está arreglado en el código (usa `spawn` automáticamente)

Si persiste:
- Usar menos workers (1-2 en lugar de 4-6)
- Usar `webui.py` en lugar de `webui_parallel.py`

### 5. Rutas de archivo mezcladas

**Problema:** Mezclar rutas Windows (C:\...) y Linux (/home/...)

**Mejores prácticas en WSL:**
```bash
# ✅ BUENO - Rutas Linux nativas
cd ~/index-tts-fork
python webui.py

# ✅ BUENO - Acceder archivos Windows vía /mnt
python inference_script.py --input /mnt/c/Users/usuario/audio.wav

# ❌ MALO - Rutas Windows directas
python inference_script.py --input C:\Users\usuario\audio.wav
```

### 6. Flash Attention build failures

**Síntoma:**
```
error: Microsoft Visual C++ 14.0 or greater is required
```

**Causa:** WSL intenta usar compilador Windows

**Solución:**
```bash
# En WSL, instalar herramientas de build Linux
sudo apt update
sudo apt install build-essential

# Build Flash Attention
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
MAX_JOBS=4 FLASH_ATTENTION_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST="10.0" python setup.py install
```

## 🎯 Rendimiento WSL vs Windows Nativo

| Aspecto | WSL2 | Windows Nativo |
|---------|------|----------------|
| **Velocidad GPU** | ≈ Igual | ≈ Igual |
| **Latencia inicial** | +10-20ms | Baseline |
| **Multiprocessing** | Más estable con spawn | Fork nativo |
| **Gestión memoria** | Compartida con Windows | Dedicada |
| **Compatibilidad** | 100% Linux tools | Limitado |
| **Flash Attention build** | ✅ Fácil | ⚠️ Complejo |

**Recomendación:** WSL2 es generalmente mejor para desarrollo y producción.

## 💡 Tips de Optimización WSL

### 1. Configuración .wslconfig óptima

```ini
# C:\Users\<usuario>\.wslconfig
[wsl2]
memory=48GB           # 75% de tu RAM total
processors=12         # 75% de tus cores
swap=8GB
localhostForwarding=true

[experimental]
autoMemoryReclaim=gradual  # Libera memoria automáticamente
```

### 2. Limitar memoria Windows para favorecer WSL

```bash
# Verificar uso de memoria
free -h

# Si WSL tiene poca memoria, ajustar .wslconfig
```

### 3. Usar distribución Ubuntu más reciente

```bash
# Listar distribuciones
wsl --list

# Instalar Ubuntu 24.04
wsl --install -d Ubuntu-24.04
```

### 4. SSD para WSL

WSL funciona mejor en SSD. Si está en HDD, moverlo:

```bash
# En PowerShell
wsl --export Ubuntu ubuntu.tar
wsl --unregister Ubuntu
wsl --import Ubuntu D:\WSL\Ubuntu ubuntu.tar
```

## 🔧 Debugging en WSL

### Script de diagnóstico

```bash
# Guarda como check_wsl.sh
#!/bin/bash

echo "=== WSL GPU Diagnostic ==="
echo ""

echo "1. WSL Version:"
cat /proc/version | grep -i microsoft
echo ""

echo "2. NVIDIA Driver:"
nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "nvidia-smi not found"
echo ""

echo "3. GPU Info:"
nvidia-smi -L 2>/dev/null || echo "No GPUs found"
echo ""

echo "4. CUDA Available in PyTorch:"
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "PyTorch not installed"
echo ""

echo "5. Memory:"
free -h
echo ""

echo "6. Disk Space:"
df -h ~
echo ""

echo "=== Done ==="
```

Ejecutar:
```bash
chmod +x check_wsl.sh
./check_wsl.sh
```

### Ver logs detallados

```bash
# Ejecutar con verbose
python webui.py --verbose 2>&1 | tee indexfts_wsl.log

# Ver solo errores
grep -i error indexfts_wsl.log
```

### Monitorear GPU en tiempo real

```bash
# Terminal 1: Ejecutar app
python webui.py

# Terminal 2: Monitorear GPU
watch -n 0.5 nvidia-smi
```

## 📚 Recursos Adicionales

### Documentación Oficial

- **WSL GPU Guide:** https://docs.nvidia.com/cuda/wsl-user-guide/index.html
- **WSL2 Install:** https://docs.microsoft.com/en-us/windows/wsl/install
- **NVIDIA Drivers:** https://www.nvidia.com/Download/index.aspx

### Troubleshooting Específico IndexTTS

- **GPU General:** Ver `GPU_TROUBLESHOOTING.md`
- **Instalación:** Ver `INSTALLATION_UPDATED.md`
- **Blackwell:** Ver `MEJORAS_BLACKWELL.md`

## ✅ Checklist Pre-Ejecución WSL

Antes de ejecutar IndexTTS en WSL, verificar:

- [ ] WSL2 instalado (no WSL1)
- [ ] Windows actualizado a última versión
- [ ] NVIDIA drivers ≥ 560.x instalados en Windows
- [ ] `nvidia-smi` funciona en WSL
- [ ] PyTorch detecta CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Suficiente RAM asignada a WSL (`.wslconfig`)
- [ ] Espacio en disco suficiente en distribución WSL
- [ ] Usando rutas Linux nativas (no rutas Windows)

## 🎯 Ejemplo Completo de Sesión WSL

```bash
# 1. Iniciar WSL
wsl

# 2. Navegar al proyecto
cd ~/index-tts-fork

# 3. Verificar GPU
nvidia-smi

# 4. Primera ejecución - configuración interactiva
python webui.py
# Seleccionar GPU, configuración se guarda

# 5. Uso normal
python webui.py
# Abre http://localhost:7860

# 6. Parallel processing
python webui_parallel.py
# Usa workers sugeridos automáticamente

# 7. Monitorear GPU (otra terminal)
watch -n 1 nvidia-smi

# 8. Al terminar (opcional)
# En PowerShell para liberar memoria:
wsl --shutdown
```

## 🏆 Ventajas de WSL2 sobre Windows Nativo

1. **Mejor compatibilidad con herramientas Linux**
   - Scripts bash nativos
   - Build tools más fáciles
   - Package managers (apt, etc.)

2. **Flash Attention más fácil de compilar**
   - GCC nativo
   - Sin necesidad de Visual Studio

3. **Multiprocessing más estable**
   - spawn configurado automáticamente
   - Menos race conditions

4. **Desarrollo más cómodo**
   - Terminal Unix-like
   - Git más rápido
   - Integración VS Code perfecta

5. **Futuro-proof**
   - Nuevas features se desarrollan primero en Linux
   - Mejor soporte de la comunidad

## 🎯 Conclusión

**IndexTTS funciona PERFECTAMENTE en WSL2** con las siguientes recomendaciones:

1. ✅ Usa WSL2 (no WSL1)
2. ✅ Drivers NVIDIA ≥ 560.x
3. ✅ Configura `.wslconfig` con suficiente memoria
4. ✅ Usa rutas Linux nativas
5. ✅ Reduce workers si hay OOM
6. ✅ Build Flash Attention desde fuente para Blackwell

**Rendimiento:** Equivalente a Linux nativo (< 5% overhead)

**Recomendación general:** WSL2 > Windows Nativo para IndexTTS
