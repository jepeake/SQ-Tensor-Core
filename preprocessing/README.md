# _Preprocessing_

_These scripts_

- _Sample Representative Weight Matrices_
- _Perform Quantisation_
- _Induce Sparsity_
- _Perform Chunking_

_to prepare the weights for the performance model._

---

### _sampler.py_

_Samples weight matrices from .safetensors files._

```bash
python sampler.py model.safetensors --num-samples 512 --verbose
python sampler.py model.safetensors -n 1024 --format numpy -o ./output
```

---

### _quantise.py_

_Quantises models using the ONNX runtime._

```bash
python quantise.py 
```

---

### _sampler_onnx.py_

_Samples weight matrices from .onnx files._

```bash
python sampler_onnx.py model.onnx --num-samples 512 --verbose
python sampler_onnx.py quantized_model.onnx -n 256 --format binary
```

---

### _chunking.py_

_Breaks sampled matrices into smaller chunks for processing by performance model._

```bash
python chunking.py sampled_weights/model_name --chunk-size 256 --save-text
python chunking.py sampled_weights/model_name --verify
```
