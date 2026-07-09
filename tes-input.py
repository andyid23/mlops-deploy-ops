import tensorflow as tf

# Load model yang sudah di-push
model_path = "C:\\Users\\Dragon\\Documents\\github\\andyid23\\mlops\\deploy-ops\\serving_model\\andyid-model\\1783556846"  # Ganti dengan versi folder yang ada
loaded_model = tf.saved_model.load(model_path)

# Cek signature
signatures = loaded_model.signatures
print("Available signatures:", signatures.keys())

# Cek input signature
if 'serving_default' in signatures:
    signature = signatures['serving_default']
    print("\nInput signature:")
    for key, value in signature.structured_input_signature[1].items():
        print(f"  {key}: {value}")