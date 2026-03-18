# Índice de Documentación del Sistema SAIS

## 📚 Descripción General de Documentación

Esta carpeta contiene análisis comprehensivo del código base de SAIS (Sistema de Inteligencia de Actividad Quirúrgica), incluyendo arquitectura, componentes y guías de implementación.

---

## 📖 Guía de Documentos

### 1. **QUICK_REFERENCE_sp.md** ⭐ COMIENCE AQUÍ
- **Mejor para**: Comenzar rápidamente, entender qué hace SAIS de un vistazo
- **Contiene**: 
  - Resumen ejecutivo (lectura de 30 seg)
  - Inicio rápido con 5 comandos
  - Flujos de trabajo comunes
  - Solución rápida de problemas
- **Tiempo de lectura**: 10-15 minutos
- **Siguiente paso**: Usar para configuración inmediata o referencia

### 2. **SETUP_CHECKLIST_sp.md** 🔧 GUÍA DETALLADA DE CONFIGURACIÓN
- **Mejor para**: Instalación paso a paso local y verificación
- **Contiene**:
  - Lista de verificación de verificación previa
  - 8 fases detalladas de configuración con ejemplos de código
  - Procedimientos de verificación después de cada paso
  - Solución de problemas para problemas comunes
  - Orientación para monitoreo de recursos
- **Tiempo de lectura**: 30 minutos de descripción general, 1-2 horas de ejecución
- **Importante**: Modificación crítica del transformador de PyTorch (Fase 2)
- **Siguiente paso**: Seguir sistemáticamente para la configuración

### 3. **SYSTEM_ANALYSIS_sp.md** 🏗️ ARQUITECTURA COMPLETA
- **Mejor para**: Comprensión profunda del diseño del sistema y componentes
- **Contiene**:
  - Resumen ejecutivo
  - Diagramas de flujo de datos de alto nivel (arte ASCII)
  - Descripciones de componentes (6 componentes principales)
  - Explicación del pipeline de flujo de datos
  - Guía completa de compilación y ejecución (80+ comandos explicados)
  - Consideraciones de desempeño
  - Parámetros de configuración
  - Gráficas de dependencia
  - Problemas comunes y soluciones
- **Tiempo de lectura**: 45 minutos (técnico)
- **Mejores capítulos**:
  - Sección 1: ¿Qué es SAIS? (no técnica)
  - Sección 2: Arquitectura del Sistema (descripción visual)
  - Sección 3: Descripción de Componentes (referencia)
  - Sección 6: Compilación y Ejecución (implementación)
- **Siguiente paso**: Referencia para entender componentes específicos

---

## 🎯 Ruta de Lectura Recomendada

### Ruta A: Quiero Ejecutar SAIS en Mi Video (Más Rápido)
1. Comience: **QUICK_REFERENCE_sp.md** → Sección "Inicio Rápido"
2. Configuración: Siga los 5 comandos
3. Problemas: Verifique la sección de Solución de Problemas
4. Atascado?: Consulte **SETUP_CHECKLIST_sp.md** para pasos detallados

### Ruta B: Quiero Entender el Sistema (Educativo)
1. Comience: **QUICK_REFERENCE_sp.md** → Secciones "¿Qué es SAIS?" + "Flujo de Datos"
2. Análisis profundo: **SYSTEM_ANALYSIS_sp.md** → Secciones 1-3 (Arquitectura)
3. Implementación: **SYSTEM_ANALYSIS_sp.md** → Sección 6 (Compilación y Ejecución)
4. Referencia: Verifique diagramas y relaciones entre componentes

### Ruta C: Quiero Configurar Todo Correctamente (Recomendado)
1. Comience: **QUICK_REFERENCE_sp.md** → Documento completo
2. Configuración: **SETUP_CHECKLIST_sp.md** → Siga Fases 1-5 exactamente
3. Prueba: Fase 6 (Prueba Rápida)
4. Ejecución: Fase 7 (Inferencia Completa)
5. Avanzado: Fase 8 (Entrenamiento) si es necesario
6. Referencia: Tenga **SYSTEM_ANALYSIS_sp.md** a mano para detalles

### Ruta D: Tengo Problemas (Depuración)
1. Verifique: **QUICK_REFERENCE_sp.md** → Sección de Solución de Problemas
2. Solución detallada: **SYSTEM_ANALYSIS_sp.md** → Sección 8 (Problemas Comunes)
3. Fase por fase: **SETUP_CHECKLIST_sp.md** → Fase Correspondiente
4. Alternativa: Ejecución manual paso a paso y verificación

---

## 📊 Guías Visuales

Tres diagramas Mermaid están incluidos en **SYSTEM_ANALYSIS_sp.md**:

1. **Diagrama de Arquitectura de Componentes**
   - Muestra: Capa de entrada → Preprocesamiento → Extracción de características → Modelo → Resultados
   - Útil: Entender flujo de datos e interacciones de componentes
   - Punto de referencia: Dónde caben DINO, RAFT, ViT

2. **Diagrama del Pipeline de Ejecución**
   - Muestra: Secuencia de ejecución de 7 pasos con requisitos previos y salidas
   - Útil: Entender qué scripts ejecutar y en qué orden
   - Punto de referencia: Validar que su ejecución coincida con el flujo esperado

3. **Diagrama de Arquitectura del Modelo**
   - Muestra: Entrada → Codificación → Modelado temporal → Clasificación → Salida
   - Útil: Entender qué sucede dentro de la red neuronal
   - Punto de referencia: Interpretar predicciones y mapas de atención

---

##  Referencia Rápida de Conceptos Clave

| Concepto | Qué Hace | Aprende Más |
|---------|----------|-----------|
| **DINO** | Transformador de visión auto-supervisado que extrae características visuales | SYSTEM_ANALYSIS_sp.md § 4.2 |
| **RAFT** | Modelo de flujo óptico que captura movimiento entre fotogramas | SYSTEM_ANALYSIS_sp.md § 4.3 |
| **ViT (Transformador de Visión)** | Modelo de aprendizaje profundo para modelado temporal | SYSTEM_ANALYSIS_sp.md § 3 |
| **Aprendizaje Contrastivo Supervisado** | Método de aprendizaje usando clasificación basada en prototipos | SYSTEM_ANALYSIS_sp.md § 4.4 |
| **Fragmento** | Ventana temporal de 5 fotogramas procesada como unidad | QUICK_REFERENCE_sp.md § Stack Técnico |
| **Mapas de Atención** | Puntuaciones de importancia del fotograma mostrando cuáles importan | SYSTEM_ANALYSIS_sp.md § Interpretabilidad |
| **Fusión Multimodal** | Combinación de características RGB + flujo óptico | SYSTEM_ANALYSIS_sp.md § 4.1 |

---

## 🔄 Resumen del Flujo de Datos

```
Entrada de Video
    ↓ [video_to_frames.sh]
Fotogramas
    ├→ Características RGB via DINO
    └→ Flujo Óptico via RAFT
         ↓
    Concatenar y Fusionar (768-dim)
         ↓ [Codificador ViT + Atención]
    Embeddings Temporales
         ↓ [Capa de Prototipos]
    Predicciones de Clase
         ↓ [Post-procesamiento]
    Predicciones Finales + Interpretabilidad
```

**Diagrama de flujo completo**: Ver SYSTEM_ANALYSIS_sp.md § Diagrama de Flujo de Datos

---

## ⚙️ Pasos Críticos de Configuración

1. **Instalar dependencias** (SETUP_CHECKLIST_sp.md § Fase 1)
   ```bash
   conda create -n SAIS python=3.9.7
   pip install -r requirements.txt
   ```

2. **Modificar transformador de PyTorch** ⚠️ (SETUP_CHECKLIST_sp.md § Fase 2)
   - Requerido para extracción de mapa de atención
   - Editar: torch/nn/modules/transformer.py
   - Líneas: 181 y 294

3. **Descargar pesos de DINO** (SETUP_CHECKLIST_sp.md § Fase 3)
   - Archivo: dino_deitsmall16_pretrain.pth (~350 MB)
   - Ubicación: SAIS/scripts/dino-main/outputs/

4. **Preparar parámetros del modelo** (SETUP_CHECKLIST_sp.md § Fase 5)
   - Ubicación: SAIS/params/Fold_0/
   - Contiene: params.zip + prototypes.zip

---

## 🚀 Comandos de Ejecución

### Mínimo (Ejecutar Inferencia)
```bash
# Supone: configuración del entorno, pesos de DINO, parámetros del modelo
cd SAIS
bash main.sh -f su_video
```

### Manual Completo
```bash
# Pasos individuales para depuración/aprendizaje
bash ./SAIS/scripts/video_to_frames.sh -f video
python ./SAIS/scripts/generate_paths.py -f video -p ./SAIS/
python ./SAIS/scripts/extract_representations.py --optical_flow [params]
python -m torch.distributed.launch ./SAIS/scripts/extract_representations.py [params]
python -m torch.distributed.launch ./SAIS/scripts/run_experiments.py --inference [params]
python ./SAIS/scripts/process_inference_results.py -p ./SAIS/
```

**Comandos completos**: Ver SYSTEM_ANALYSIS_sp.md § Paso 6

---

## 📁 Estructura del Repositorio

```
SAIS/
├── README.md                    # README original del proyecto
├── QUICK_REFERENCE_sp.md        # Esta documentación
├── SETUP_CHECKLIST_sp.md        # Configuración paso a paso
├── SYSTEM_ANALYSIS_sp.md        # Análisis comprehensivo
│
├── SAIS/                        # Paquete principal
│   ├── main.sh                  # Punto de entrada (¡ejecute esto!)
│   ├── __init__.py
│   └── scripts/
│       ├── run_experiments.py   # Orquestador (entrenamiento/inferencia)
│       ├── train.py             # Coordinador de entrenamiento
│       ├── extract_representations.py  # Extracción de características
│       ├── generate_paths.py    # Generación de rutas
│       ├── prepare_model.py     # Constructor de modelo
│       ├── prepare_dataset.py   # Cargador de datos
│       ├── prepare_miscellaneous.py    # Utilidades
│       ├── perform_training.py  # Bucle de entrenamiento
│       ├── process_inference_results.py # Post-procesamiento
│       ├── video_to_frames.sh   # Extracción de fotogramas
│       └── dino-main/           # Extractor de características (DINO)
│
├── videos/                      # ENTRADA: Coloque sus videos aquí
├── params/                      # Parámetros del modelo (si está disponible)
│   └── Fold_0/
│       ├── params.zip
│       └── prototypes.zip
│
├── images/                      # Generado: Fotogramas extraídos
├── flows/                       # Generado: Flujo óptico
├── paths/                       # Generado: Archivos CSV de rutas
├── results/                     # Generado: Características extraídas
└── predictions/                 # SALIDA: Predicciones finales
```

---

## 🎓 Recursos de Aprendizaje

### Teoría
- **Artículo Original**: [SAIS - Nature Biomedical Engineering](https://www.nature.com/articles/s41551-023-01010-8)
- **Transformadores de Visión**: [Artículo ViT](https://arxiv.org/abs/2010.11929)
- **DINO**: [ViT Auto-supervisado](https://arxiv.org/abs/2104.14294)
- **Aprendizaje Contrastivo**: [SimCLR](https://arxiv.org/abs/2002.05709)

### Referencias de Código
- **Repositorio DINO**: https://github.com/facebookresearch/dino
- **Documentación de PyTorch**: https://pytorch.org/docs/
- **Flujo Óptico RAFT**: https://github.com/princeton-vl/RAFT

---

## 🐛 Enlaces Rápidos de Problemas Comunes

| Problema | Documento | Sección |
|-------|----------|---------|
| "ModuleNotFoundError" | SYSTEM_ANALYSIS_sp.md | Problemas Comunes |
| CUDA sin memoria | QUICK_REFERENCE_sp.md | Solución de Problemas |
| Error de transformador PyTorch | SETUP_CHECKLIST_sp.md | Fase 2 |
| Pesos de DINO faltantes | SETUP_CHECKLIST_sp.md | Fase 3 |
| No se generan predicciones | QUICK_REFERENCE_sp.md | Solución de Problemas |
| Video no detectado | QUICK_REFERENCE_sp.md | Solución de Problemas |

---

## ✅ Listas de Verificación de Validación

### Después de la Instalación
- [ ] Entorno Conda creado y activado
- [ ] Todos los paquetes instalados (torch 1.8.0, torchvision 0.9.0, etc.)
- [ ] Transformador de PyTorch modificado (retorna atención)
- [ ] CUDA disponible y accesible
- [ ] Pesos de DINO descargados (~350 MB)

### Antes de la Primera Ejecución
- [ ] Video colocado en SAIS/videos/
- [ ] Parámetros del modelo en SAIS/params/Fold_0/
- [ ] Espacio en disco suficiente (~100 GB mínimo)
- [ ] GPU disponible (verificar: nvidia-smi)

### Después de la Primera Ejecución
- [ ] La carpeta images/ contiene fotogramas extraídos
- [ ] La carpeta flows/ contiene flujo óptico
- [ ] La carpeta results/ contiene archivos de características HDF5
- [ ] La carpeta predictions/ contiene resultados CSV
- [ ] Sin mensajes de error en la consola

---

## 📞 Obtener Ayuda

1. **Profundidad**: Verifique SYSTEM_ANALYSIS_sp.md § 8 (Problemas Comunes y Soluciones)
2. **Corrección rápida**: Verifique QUICK_REFERENCE_sp.md § Solución de Problemas
3. **Problemas de configuración**: Verifique SETUP_CHECKLIST_sp.md Fase Correspondiente
4. **Oficial**: Visite [SAIS GitHub](https://github.com/danikiyasseh/SAIS)
5. **Citación/Contacto**: Ver README del repositorio

---

## 📝 Mantenimiento de Documentación

**Última Actualización**: 2024
**Versión de Documentación**: 1.0
**Versión de SAIS Referenciada**: Implementación de autores del artículo original

Si encuentra errores de documentación o tiene mejoras:
1. Verifique contra el repositorio oficial de SAIS
2. Reporte problemas a los mantenedores del repositorio
3. Actualice copias locales con correcciones

---

## 🎯 Navegación Rápida

**Quiero...**
- ✅ Comenzar inmediatamente → **QUICK_REFERENCE_sp.md**
- ✅ Configurar en mi máquina → **SETUP_CHECKLIST_sp.md**
- ✅ Entender la arquitectura → **SYSTEM_ANALYSIS_sp.md**
- ✅ Encontrar un componente específico → Use tabla de contenido en cada documento
- ✅ Depurar un problema → Busque el problema en secciones "Solución de Problemas"
- ✅ Aprender qué hace cada archivo → **Descripción de Componentes** en SYSTEM_ANALYSIS_sp.md

---

## 📊 Estadísticas de Documentos

| Documento | Longitud | Tiempo de Lectura | Mejor Para |
|----------|---------|--------------|----------|
| QUICK_REFERENCE_sp.md | ~3,000 palabras | 10-15 min | Inicio rápido, descripción general |
| SETUP_CHECKLIST_sp.md | ~4,500 palabras | 30 min lectura, 1-2 horas ejecución | Configuración detallada |
| SYSTEM_ANALYSIS_sp.md | ~6,500 palabras | 45 min lectura | Comprensión profunda |
| **Total** | **~14,000 palabras** | **~1.5 horas lectura** | **Cobertura completa** |

---

**¡Feliz análisis de videos quirúrgicos! 🎥🔬**
