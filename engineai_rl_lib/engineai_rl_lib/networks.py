import torch.nn as nn


class CombinedNetworks(nn.Module):
    def __init__(self, networks_dict):
        super().__init__()
        self.network_dicts = networks_dict
        self.networks = nn.ModuleList([network["network"] for network in networks_dict])

    def forward(self, inputs):
        outputs = []
        output_names = []
        last_input_names = []
        for idx, (network_dict, network) in enumerate(
            zip(self.network_dicts, self.networks)
        ):
            if idx != 0:
                inputs = self.get_input_from_last_output(
                    output_names,
                    network_dict["forward_inputs"],
                    last_input_names,
                    inputs,
                    outputs,
                )
            else:
                separated_inputs = []
                idx = 0
                for input_name in network_dict["forward_inputs"]:
                    separated_inputs.append(
                        inputs[
                            idx : idx + network_dict["forward_input_dims"][input_name]
                        ]
                    )
                    idx += network_dict["forward_input_dims"][input_name]
                inputs = separated_inputs

            outputs = network(*inputs)
            output_names = network_dict["forward_outputs"]
            last_input_names = network_dict["forward_inputs"]
        return outputs

    def get_input_from_last_output(
        self, output_names, input_names, last_input_names, last_inputs, outputs
    ):
        inputs = []
        for input_name in input_names:
            if input_name in last_input_names:
                inputs.append(last_inputs[last_input_names.index(input_name)])
            elif input_name in output_names:
                inputs.append(outputs[output_names.index(input_name)])
            else:
                raise NameError(f"Input: {input_name} doesn't exist")
        return inputs
