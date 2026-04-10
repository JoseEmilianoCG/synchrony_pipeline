# EEG Synchrony Pipeline

Pipeline de procesamiento y análisis de señales EEG enfocado en la detección de sincronización inter-cerebral durante actividades de lectura.

## Descripción

Este proyecto implementa un pipeline completo para procesar señales de electroencefalografía (EEG), extraer métricas de sincronía entre individuos y entrenar modelos de machine learning capaces de distinguir entre distintas condiciones cognitivas (reposo, lectura individual y lectura compartida).

El sistema abarca desde la carga de datos crudos hasta la generación automática de reportes con resultados visuales e interpretables.

## Problema

El proyecto busca responder si existen patrones de sincronización cerebral que permitan diferenciar entre distintos niveles de actividad cognitiva durante la lectura.

La sincronización inter-cerebral es un fenómeno clave en neurociencia social, asociado a procesos como la atención conjunta y la interacción colaborativa :contentReference[oaicite:0]{index=0}.

## Datos

- Fuente: Experimento realizado en la FIL Monterrey 2025 :contentReference[oaicite:1]{index=1}  
- Participantes: 20 (10 adultos, 10 niños)
- Condiciones:
  - Reposo
  - Lectura individual
  - Lectura compartida
- Dispositivo: MUSE (4 electrodos)
- Segmentación: ventanas de 2 segundos

## Pipeline general

El flujo del sistema es:

1. **Carga y procesamiento de datos (C1)**
2. **Entrenamiento de modelos de ML (C4)**
3. **Optimización de hiperparámetros (C3)**
4. **Evaluación final**
5. **Visualización y generación de reporte (C6)**

Este flujo está orquestado desde el script principal:

`scripts/main.py` :contentReference[oaicite:2]{index=2}

## Estructura de repositorio

synchrony_pipeline/
├── configs/                 # Configuraciones del sistema
│   └── defaults.yaml
│
├── data/
│   ├── rawdata/             # Datos crudos
│   ├── interim/             # Datos intermedios
│   └── processed/           # Datos procesados
│
├── docs/                    # Documentación
│
├── reports/
│   └── figures/             # Resultados y visualizaciones
│
├── scripts/                 # Scripts ejecutables principales
│   └── main.py
│
├── src/                     # Código fuente principal
│   ├── data/                # Carga y preprocesamiento
│   ├── features/            # Extracción de características
│   ├── models/              # Modelos y entrenamiento
│   ├── optimization/        # Optimización / DoE
│   └── visualization/       # Visualización de resultados
│
├── tests/                   # Tests
│
├── requirements.txt
└── README.md

## Flujo de ejecución

```bash
python scripts/main.py


