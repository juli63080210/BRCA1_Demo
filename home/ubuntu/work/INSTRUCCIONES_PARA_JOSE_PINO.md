# Demo Local del Sistema de Diagnóstico Genético BRCA1

Estimado José,

Este paquete contiene una demo funcional del sistema de diagnóstico genético BRCA1 detección temprana de mutaciones genéticas asociadas a enfermedades complejas como el cáncer. El gen BRCA1, por su relevancia en el cáncer de mama y ovario hereditario,lo que se busca es despues ir agregando mas mutaciones mas datos mejor lectura avalar esto con expertos pero este ha sido nuestro punto de partida estratégico basada en IA, diseñada para ser ejecutada localmente. El objetivo es proporcionarte una visión transparente del proceso de inferencia, permitiéndote observar cómo el sistema procesa las secuencias genéticas y llega a sus conclusiones, con especial énfasis en la interacción entre H-Net y Turbit.

## Contenido del Paquete:

-   **Modelos Entrenados:** `modelo_hnet_improved.pt`, `modelo_turbit_conv.pt`, `umbral_anomalia.pt`
-   **Arquitecturas:** `hnet_lite_improved.py`, `turbit_conv_autoencoder.py`, `preprocess_sequence.py`
-   **Script Principal:** `predict_with_poc.py` (modificado para visualización detallada del proceso)
-   **Archivos FASTA de Prueba:** `brca1_variante_1.fasta`, `brca1_variante_3.fasta`, `brca1_reference.fasta`, `brca1_mutated_variant_for_demo.fasta`
-   **Documentación:** `VERIFICACION_FUNCIONAMIENTO.md` (resultados esperados)

## Requisitos y Configuración:

-   **Python 3.11+**
-   **Dependencias:** Instala vía pip:
    ```bash
    pip install torch scikit-learn biopython numpy
    ```

## Ejecución de la Demo:

1.  **Descomprime el paquete** en tu directorio de trabajo.
2.  **Navega al directorio** de la demo en tu terminal.
3.  **Ejecuta el script principal:**
    ```bash
    python predict_with_poc.py
    ```

### Observación del Proceso (Transparencia):

El script `predict_with_poc.py` ha sido instrumentado para imprimir en la consola cada etapa del análisis:

-   **Análisis de Secuencia:** Muestra la codificación numérica de la secuencia de entrada.
-   **Predicción H-Net:** Detalla la salida cruda del modelo, la probabilidad de mutación (sigmoid) y la predicción inicial (SANA/MUTADA).
-   **Detección de Anomalías (Turbit):** Presenta el error de reconstrucción (MSE), el umbral de anomalía y el resultado de la detección de anomalías.
-   **Lógica de Decisión Final:** Explica cómo se combinan las predicciones de H-Net y Turbit para llegar al diagnóstico final, incluyendo casos donde se recomienda revisión manual.

### Prueba con Diferentes Secuencias:

Por defecto, `predict_with_poc.py` analiza `brca1_variante_1.fasta`. Para probar otras secuencias, edita la línea `ruta_variante = "..."` en el script y re-ejecuta. Los archivos de prueba incluidos (`brca1_reference.fasta`, `brca1_mutated_variant_for_demo.fasta`, `brca1_variante_3.fasta`) te permitirán explorar diferentes escenarios.

## Verificación del Funcionamiento:

Consulta `VERIFICACION_FUNCIONAMIENTO.md` para los resultados esperados de cada archivo FASTA de prueba. Esto te permitirá validar que el sistema se comporta como se describe.

Espero que esta demo te sea útil para una comprensión más profunda del sistema. Cualquier pregunta o feedback es bienvenido.

