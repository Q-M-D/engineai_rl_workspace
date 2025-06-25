import os
import torch
import copy


def convert_nn_to_onnx(model, path, name):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, name + ".onnx")
    model.eval()

    dummy_inputs = tuple(
        torch.randn(input_dim)
        for _, input_dim in model.network_dicts[0]["forward_input_dims"].items()
    )
    if len(dummy_inputs) > 1:
        dummy_inputs = torch.cat(dummy_inputs, dim=-1)
    torch.onnx.export(
        model,
        dummy_inputs,
        path,
        verbose=True,
        export_params=True,
        opset_version=13,
    )
    print("Exported policy as onnx script to: ", path)


def convert_onnx_to_mnn(onnx_path, mnn_path):
    os.system(
        "mnnconvert -f ONNX --modelFile "
        + onnx_path
        + " --MNNModel "
        + mnn_path
        + " --bizCode biz"
    )
    if ".__convert_external_data.bin" in os.listdir("."):
        os.remove(".__convert_external_data.bin")


def export_policy_as_jit(actor_critic, path):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "policy_1.pt")
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path)
