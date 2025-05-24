from transformers import ViTForImageClassification, ViTImageProcessor
from optimum.onnxruntime import ORTModelForImageClassification, ORTQuantizer
from optimum.onnxruntime.configuration import QuantizationConfig
import os
import numpy as np


#     ██████                                     █████     ███                  
#   ███░░░░███                                  ░░███     ░░░                   
#  ███    ░░███ █████ ████  ██████   ████████   ███████   ████   █████   ██████ 
# ░███     ░███░░███ ░███  ░░░░░███ ░░███░░███ ░░░███░   ░░███  ███░░   ███░░███
# ░███   ██░███ ░███ ░███   ███████  ░███ ░███   ░███     ░███ ░░█████ ░███████ 
# ░░███ ░░████  ░███ ░███  ███░░███  ░███ ░███   ░███ ███ ░███  ░░░░███░███░░░  
#  ░░░██████░██ ░░████████░░████████ ████ █████  ░░█████  █████ ██████ ░░██████ 
#    ░░░░░░ ░░   ░░░░░░░░  ░░░░░░░░ ░░░░ ░░░░░    ░░░░░  ░░░░░ ░░░░░░   ░░░░░░  

# Script Quantises with ONNX


model_name = "google/vit-base-patch16-224"

ort_model = ORTModelForImageClassification.from_pretrained(
    model_name,
    export=True
)

onnx_model_path = "./vit_onnx"
ort_model.save_pretrained(onnx_model_path)
print(f"Model Exported to ONNX: {onnx_model_path}")

quantization_config = QuantizationConfig(
    is_static=False,                    
    format="QOperator",                
    mode="IntegerOps",                  
    activations_dtype="QUInt8",         
    weights_dtype="QUInt4",             
    per_channel=False,                 
    reduce_range=True,                
)
quantizer = ORTQuantizer.from_pretrained(onnx_model_path)

print("Quantising to INT4")
output_dir = "./vit_int4_quantized"

quantizer.quantize(
    quantization_config=quantization_config,
    save_dir=output_dir,
    file_suffix="quantized"
)
print(f"Quantised Model Saved: {output_dir}")

quantized_model = ORTModelForImageClassification.from_pretrained(output_dir)

print("Testing Inference")
processor = ViTImageProcessor.from_pretrained(model_name)

dummy_input = np.random.rand(224, 224, 3).astype(np.uint8)
inputs = processor(dummy_input, return_tensors="pt")

outputs = quantized_model(**inputs)
logits = outputs.logits

print(f"Inference Successful. Output Shape: {logits.shape}")
print(f"Output Range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")

print(f"Quantised Model Ready at: {os.path.abspath(output_dir)}")
    