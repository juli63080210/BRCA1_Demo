# BRCA1 Demo: Sistema de Diagnóstico Genético Basado en IA

Este proyecto presenta una demostración funcional del sistema de diagnóstico genético para el gen BRCA1, impulsado por inteligencia artificial. Su objetivo principal es proporcionar una visión transparente y detallada del proceso de inferencia, permitiendo a los usuarios comprender cómo el sistema analiza las secuencias genéticas y llega a sus conclusiones. Se pone un énfasis particular en la interacción y el flujo de trabajo entre los modelos H-Net y Turbit.

## Contenido del Repositorio:

-   **Modelos Entrenados:** `modelo_hnet_improved.pt`, `modelo_turbit_conv.pt`, `umbral_anomalia.pt` (Modelos pre-entrenados para la detección de mutaciones y anomalías).
-   **Arquitecturas:** `hnet_lite_improved.py`, `turbit_conv_autoencoder.py`, `preprocess_sequence.py` (Definiciones de las arquitecturas de red neuronal y scripts de preprocesamiento de datos).
-   **Script Principal:** `predict_with_poc.py` (Script modificado para la visualización detallada del proceso de inferencia, mostrando cada etapa del análisis).
-   **Archivos FASTA de Prueba:** `brca1_variante_1.fasta`, `brca1_variante_3.fasta`, `brca1_reference.fasta`, `brca1_mutated_variant_for_demo.fasta` (Ejemplos de secuencias genéticas para probar el sistema).
-   **Documentación Adicional:** `VERIFICACION_FUNCIONAMIENTO.md` (Documento que describe los resultados esperados para cada archivo FASTA de prueba, permitiendo la validación del sistema).
-   **Instrucciones de Uso:** `instrucciones_para_uso.md` (Guía detallada para la ejecución y comprensión de la demo).

## Requisitos y Configuración:

Para ejecutar esta demo localmente, necesitarás:

-   **Python 3.11+**
-   **Dependencias:** Instala las librerías necesarias utilizando `pip`:
    ```bash
    pip install torch scikit-learn biopython numpy
    ```
    (Un archivo `requirements.txt` con estas dependencias también está incluido para facilitar la instalación: `pip install -r requirements.txt`)

## Ejecución de la Demo:

1.  **Clona este repositorio** en tu máquina local.
2.  **Navega al directorio** del proyecto en tu terminal.
3.  **Instala las dependencias** (si aún no lo has hecho):
    ```bash
    pip install -r requirements.txt
    ```
4.  **Ejecuta el script principal:**
    ```bash
    python predict_with_poc.py
    ```

## Observación Detallada del Proceso (Transparencia y Explicabilidad):

El script `predict_with_poc.py` ha sido instrumentado para ofrecer una visibilidad completa de cada etapa del análisis, lo cual es crucial para entender el funcionamiento interno del sistema:

-   **Análisis de Secuencia:** Muestra la codificación numérica de la secuencia de entrada, explicando cómo los datos genéticos brutos se transforman en un formato procesable por los modelos de IA.
-   **Predicción H-Net:** Detalla la salida cruda del modelo H-Net, la probabilidad de mutación calculada (usando una función sigmoide) y la predicción inicial de si la secuencia es SANA o MUTADA. Esto permite ver la confianza del modelo en su clasificación.
-   **Detección de Anomalías (Turbit):** Presenta el error de reconstrucción (Mean Squared Error - MSE) generado por el autoencoder Turbit, el umbral de anomalía predefinido y el resultado de la detección de anomalías. Un MSE alto en relación con el umbral indica una secuencia atípica o anómala.
-   **Lógica de Decisión Final:** Explica cómo se combinan las predicciones de H-Net y Turbit para llegar al diagnóstico final. Se detallan los casos en los que se recomienda una revisión manual por parte de un experto, especialmente cuando las predicciones de ambos modelos son contradictorias o cuando la anomalía detectada es significativa pero la mutación no es clara.

### Prueba con Diferentes Secuencias:

Por defecto, `predict_with_poc.py` analiza `brca1_variante_1.fasta`. Para probar otras secuencias, puedes editar la línea `ruta_variante = "..."` dentro del script `predict_with_poc.py` y re-ejecutarlo. Los archivos de prueba incluidos (`brca1_reference.fasta`, `brca1_mutated_variant_for_demo.fasta`, `brca1_variante_3.fasta`) te permitirán explorar diversos escenarios, incluyendo variantes de referencia, mutadas y otras variantes.

## Problemas Actuales y Áreas de Mejora (Visión a Futuro):

Este proyecto es una prueba de concepto y, como tal, presenta ciertas limitaciones que son oportunidades para futuras mejoras y colaboración:

-   **Alcance Limitado:** Actualmente, el sistema se enfoca exclusivamente en el gen BRCA1 y en la detección de mutaciones puntuales o pequeñas inserciones/deleciones. La meta a largo plazo es expandir su capacidad para analizar el genoma humano completo, lo cual requerirá modelos mucho más complejos y eficientes.
-   **Generalización:** Los modelos actuales han sido entrenados con un conjunto de datos específico. Mejorar la generalización a una gama más amplia de variaciones genéticas y poblaciones es crucial. Esto implica la necesidad de conjuntos de datos más diversos y técnicas de entrenamiento más robustas.
-   **Validación Clínica:** La validación actual se basa en pruebas de funcionamiento con secuencias conocidas. Para un uso clínico real, se requiere una validación rigurosa con datos de pacientes reales y la supervisión de expertos en genética y medicina.
-   **Interpretación de Anomalías:** Aunque Turbit detecta anomalías, la interpretación biológica de estas anomalías aún requiere un análisis humano. Se podría desarrollar un módulo para proporcionar explicaciones más detalladas sobre la naturaleza de la anomalía detectada.
-   **Rendimiento y Escalabilidad:** A medida que el alcance se expanda al genoma completo, el rendimiento computacional será un desafío. La optimización de los modelos y el uso de arquitecturas más eficientes o hardware especializado serán necesarios.
-   **Interfaz de Usuario:** Actualmente, la interacción es a través de la línea de comandos. Una interfaz gráfica de usuario (GUI) o una API web facilitarían enormemente su uso por parte de no-expertos.

## Contribución y Colaboración:

Este proyecto está bajo la Licencia MIT, lo que fomenta la colaboración abierta. ¡Tus contribuciones son bienvenidas! Si estás interesado en ayudar a evolucionar este sistema hacia el análisis de todo el genoma humano o en abordar cualquiera de los problemas actuales, no dudes en:

-   Abrir un `Issue` para reportar errores o sugerir nuevas características.
-   Enviar un `Pull Request` con tus mejoras o nuevas funcionalidades.

Tu autoría será siempre reconocida en cualquier contribución significativa. ¡Juntos podemos hacer que este proyecto sea una herramienta invaluable para el diagnóstico genético!.
