import torch
import torch.nn as nn
from hnet_lite_improved import HNetLiteImproved
from turbit_conv_autoencoder import TurbitConvAutoencoder
from preprocess_sequence import preprocess_fasta_for_model

def predict_sequence(fasta_path):
    # Cargar modelos y umbral
    hnet = HNetLiteImproved()
    hnet.load_state_dict(torch.load("modelo_hnet_improved.pt"))
    hnet.eval()

    turbit = TurbitConvAutoencoder()
    turbit.load_state_dict(torch.load("modelo_turbit_conv.pt"))
    turbit.eval()

    umbral_anomalia = torch.load("umbral_anomalia.pt")

    # Leer y codificar la secuencia del archivo FASTA usando la nueva función de preprocesamiento
    seq_tensor = preprocess_fasta_for_model(fasta_path, max_len=7500)
    print(f"\n--- Análisis de Secuencia: {fasta_path} ---")
    print(f"Secuencia codificada (primeros 10 bases):\n{seq_tensor[0, :10, :].numpy()}")

    # Predicción con H-Net
    with torch.no_grad():
        hnet_output = hnet(seq_tensor).squeeze()
        prob_mutacion = torch.sigmoid(hnet_output).item()
        pred_hnet = "MUTADA" if prob_mutacion > 0.5 else "SANA"
    print(f"\n--- Predicción H-Net ---")
    print(f"Salida cruda de H-Net: {hnet_output.item():.4f}")
    print(f"Probabilidad de Mutación (Sigmoid): {prob_mutacion:.4f}")
    print(f"Predicción H-Net: {pred_hnet}")

    # Detección de anomalías con Turbit
    with torch.no_grad():
        reconstruccion = turbit(seq_tensor)
        error_reconstruccion = torch.mean((reconstruccion - seq_tensor) ** 2).item()
        es_anomalia = error_reconstruccion > umbral_anomalia.item()
    print(f"\n--- Detección de Anomalías (Turbit) ---")
    print(f"Error de Reconstrucción (MSE): {error_reconstruccion:.6f}")
    print(f"Umbral de Anomalía: {umbral_anomalia.item():.6f}")
    print(f"Anomalía Detectada: {"Sí" if es_anomalia else "No"}")

    # Generar resultado
    resultado = {
        "Archivo": fasta_path,
        "Predicción H-Net": pred_hnet,
        "Probabilidad de Mutación": f"{prob_mutacion:.4f}",
        "Error de Reconstrucción (Turbit)": f"{error_reconstruccion:.6f}",
        "Umbral de Anomalía": f"{umbral_anomalia.item():.6f}",
        "Anomalía Detectada": "Sí" if es_anomalia else "No"
    }

    # Lógica de decisión final
    print(f"\n--- Lógica de Decisión Final ---")
    if es_anomalia and pred_hnet == "SANA":
        resultado["Diagnóstico Final"] = "Anomalía detectada en secuencia clasificada como SANA. Revisión manual recomendada."
        print("H-Net clasificó como SANA, pero Turbit detectó una anomalía. Se recomienda revisión manual.")
    elif es_anomalia and pred_hnet == "MUTADA":
        resultado["Diagnóstico Final"] = "MUTADA (con alta confianza, anomalía consistente con mutación)"
        print("H-Net clasificó como MUTADA y Turbit detectó una anomalía. Alta confianza en la mutación.")
    else:
        resultado["Diagnóstico Final"] = pred_hnet
        print(f"Diagnóstico final basado en H-Net: {pred_hnet}")

    return resultado

if __name__ == "__main__":
    # Prueba con una de las variantes proporcionadas
    # (Asegúrate de que la ruta al archivo FASTA sea correcta)
    ruta_variante = "brca1_variante_1.fasta"
    resultado_prediccion = predict_sequence(ruta_variante)
    
    print("--- Resultado de la Predicción para la Prueba de Concepto ---")
    for key, value in resultado_prediccion.items():
        print(f"{key}: {value}")

