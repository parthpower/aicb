import torch
from apex.contrib.layer_norm.layer_norm import FastLayerNormFN
from utils.utils import *
import random

def main():

    @cuda_timing_decorator
    def _apply_fused_layer_norm(hidden_states, lay_weight, bias):
        output_lay = FastLayerNormFN.apply(
            hidden_states, lay_weight, bias, 1e-05
        )
        return output_lay

    @cuda_timing_decorator
    def _apply_torch_layernorm(hidden_states, lay_weight, bias):
        output_lay = torch.nn.functional.layer_norm(
            hidden_states, [x.shape[0]] , lay_weight, bias, 1e-05
        )
        return output_lay

    itr = 10
    # sizes as per megatron_gpt.sh
    hidden_sizes = [12288, 6144, 5120, 4096, 16384, 8192]
    hidden_sizes.sort()
    for h in hidden_sizes:
        x = torch.rand(h).to(torch.bfloat16)
        l = torch.rand(h).to(torch.bfloat16)
        b = torch.zeros(h).to(torch.bfloat16)
        t_torch_total = 0
        t_fused_total = 0
        norm_diff = 0
        # ignore first run because of kernel loading
        _apply_torch_layernorm(x, l, b)
        _apply_fused_layer_norm(x, l, b)
        for i in range(itr):
            y_torch, t_torch = _apply_torch_layernorm(x, l, b)
            y_fused, t_fused = _apply_fused_layer_norm(x, l, b)
            norm_diff += torch.norm(y_torch - y_fused)
            t_torch_total += t_torch
            t_fused_total += t_fused
        diff = (t_torch_total - t_fused_total)/(t_fused_total) * 100
        t_torch_total = t_torch_total/float(itr)
        t_fused_total = t_fused_total/float(itr)
        nrom_diff = norm_diff/float(itr)
        print(f"hidden: {h} t_fused: {t_fused_total:.2f} t_torch: {t_torch_total:.2f} diff: {diff:.2f}% norm_diff: {norm_diff:.2f}")

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    random.seed(0)

    main()

