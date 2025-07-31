# Verificación del Funcionamiento de la Demo Local

Este documento describe los resultados esperados al ejecutar `predict_with_poc.py` con los archivos FASTA de prueba incluidos. Dada tu experiencia con Turbit, esta sección se enfoca en los resultados clave para validar el comportamiento del sistema.

## Resultados Esperados por Archivo FASTA

### 1. `brca1_variante_1.fasta` (Secuencia Mutada Conocida)

**Comportamiento Esperado:** El sistema debe clasificar esta secuencia como **MUTADA** y detectar una **anomalía** consistente con la mutación.

**Resultados Clave en la Salida de Consola:**
-   `Predicción H-Net: MUTADA`
-   `Probabilidad de Mutación: > 0.5` (ej. `0.5464`)
-   `Error de Reconstrucción (Turbit): > Umbral de Anomalía` (ej. `0.181094` vs `0.180383`)
-   `Anomalía Detectada: Sí`
-   `Diagnóstico Final: MUTADA (con alta confianza, anomalía consistente con mutación)`

### 2. `brca1_reference.fasta` (Secuencia de Referencia Sana)

**Comportamiento Esperado:** El sistema debe clasificar esta secuencia como **SANA** y **no detectar anomalías**.

**Resultados Clave en la Salida de Consola (modificando `ruta_variante` en `predict_with_poc.py`):**
-   `Predicción H-Net: SANA`
-   `Probabilidad de Mutación: < 0.5`
-   `Error de Reconstrucción (Turbit): < Umbral de Anomalía`
-   `Anomalía Detectada: No`
-   `Diagnóstico Final: SANA`

### 3. `brca1_mutated_variant_for_demo.fasta` (Secuencia Mutada Generada)

**Comportamiento Esperado:** Similar a `brca1_variante_1.fasta`, debe clasificarse como **MUTADA** con **anomalía**.

**Resultados Clave en la Salida de Consola (modificando `ruta_variante` en `predict_with_poc.py`):**
-   `Predicción H-Net: MUTADA`
-   `Probabilidad de Mutación: > 0.5`
-   `Error de Reconstrucción (Turbit): > Umbral de Anomalía`
-   `Anomalía Detectada: Sí`
-   `Diagnóstico Final: MUTADA (con alta confianza, anomalía consistente con mutación)`

## Solución de Problemas:

Si los resultados no coinciden con lo esperado, verifica:

-   Instalación de dependencias (`pip install ...`).
-   Versión de Python (3.11+).
-   Integridad de los archivos del paquete.
-   Cualquier error en la salida de la terminal.

