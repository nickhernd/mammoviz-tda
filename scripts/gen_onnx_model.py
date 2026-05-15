"""
Generate a minimal ONNX model that mimics a breast cancer CNN classifier.
Input:  image_features (float32, [1, 512]) + tda_features (float32, [1, 192])
Output: logits (float32, [1, 2])  — benign / malignant

The weights are random; the point is a valid ONNX graph that ONNX Runtime can run.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import os

OUT = "data/models/breast_cnn.onnx"
os.makedirs("data/models", exist_ok=True)

rng = np.random.default_rng(0)

IMG_DIM = 512
TDA_DIM = 192   # 3 dims × 64 bins
HIDDEN  = 128
OUT_DIM = 2

# Weights
W1 = rng.normal(0, 0.02, (IMG_DIM + TDA_DIM, HIDDEN)).astype(np.float32)
b1 = np.zeros(HIDDEN, dtype=np.float32)
W2 = rng.normal(0, 0.02, (HIDDEN, OUT_DIM)).astype(np.float32)
b2 = np.zeros(OUT_DIM, dtype=np.float32)

def make_tensor(name, arr):
    return numpy_helper.from_array(arr, name=name)

nodes = [
    helper.make_node("Concat",     ["image_features", "tda_features"], ["concat"], axis=1),
    helper.make_node("MatMul",     ["concat", "W1"],                   ["mm1"]),
    helper.make_node("Add",        ["mm1", "b1"],                      ["add1"]),
    helper.make_node("Relu",       ["add1"],                           ["relu1"]),
    helper.make_node("MatMul",     ["relu1", "W2"],                    ["mm2"]),
    helper.make_node("Add",        ["mm2", "b2"],                      ["logits"]),
]

graph = helper.make_graph(
    nodes,
    "breast_cnn",
    inputs=[
        helper.make_tensor_value_info("image_features", TensorProto.FLOAT, [1, IMG_DIM]),
        helper.make_tensor_value_info("tda_features",   TensorProto.FLOAT, [1, TDA_DIM]),
    ],
    outputs=[
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, OUT_DIM]),
    ],
    initializer=[
        make_tensor("W1", W1),
        make_tensor("b1", b1),
        make_tensor("W2", W2),
        make_tensor("b2", b2),
    ]
)

model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, OUT)
print(f"ONNX model saved to '{OUT}'  (inputs: image_features[1,{IMG_DIM}] + tda_features[1,{TDA_DIM}]  →  logits[1,{OUT_DIM}])")
