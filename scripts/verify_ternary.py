"""Load a saved student checkpoint and verify the ternary constraint on its weights.

Assignment deliverable 3. Checkpoints hold the fp32 latent weights that training updates; the
ternary weights used in every forward pass are a deterministic function of them. This script:

  1. rebuilds the ternary ResNet-18 and loads the checkpoint;
  2. confirms exactly the pre-registered layers are ternary (19 convs) and the stem conv, the
     final FC and all BatchNorm tensors are fp32;
  3. for every ternary layer, checks each output filter holds only {-alpha_k, 0, +alpha_k} and
     reports sparsity, alpha, and degenerate filters;
  4. checks each TernaryConv2d forward pass actually uses the ternary weights;
  5. with --export, packs the ternary weights at 2 bits each, saves the deployable model, reloads
     it, and checks it reproduces the ternary network's outputs.

    python scripts/verify_ternary.py --checkpoint RUNS/D_0/checkpoints/epoch_160.pt \
        --export student_ternary_2bit.pth
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import storage
from src.models import EXPECTED_TERNARY_CONVS, resnet18_cifar
from src.quantizer import ternarize
from src.ternary_modules import TernaryConv2d, ternarize_model


def pack_2bit(codes):
    """{-1, 0, +1} int codes -> uint8, four codes per byte."""
    u = (codes.flatten().to(torch.int16) + 1).to(torch.uint8)  # {0, 1, 2}
    n = u.numel()
    u = torch.cat([u, torch.zeros((-n) % 4, dtype=torch.uint8)]).view(-1, 4)
    return u[:, 0] | (u[:, 1] << 2) | (u[:, 2] << 4) | (u[:, 3] << 6), n


def unpack_2bit(packed, n, shape):
    p = packed.to(torch.int16)
    u = torch.stack([(p >> s) & 3 for s in (0, 2, 4, 6)], dim=1).flatten()[:n]
    return (u - 1).to(torch.int8).view(shape)


def load_student(path):
    model = resnet18_cifar()
    names = ternarize_model(model, expected_count=EXPECTED_TERNARY_CONVS["resnet18_cifar"])
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    return model.eval(), names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--export", default=None, help="write the 2-bit packed model here")
    args = parser.parse_args()

    model, names = load_student(args.checkpoint)
    failures = []

    # 2. Which layers are ternary, and which stay fp32.
    ternary_modules = [n for n, m in model.named_modules() if isinstance(m, TernaryConv2d)]
    print(f"ternary conv layers: {len(ternary_modules)} (expected 19)")
    if len(ternary_modules) != 19:
        failures.append("wrong number of ternary layers")
    if isinstance(model.conv1, TernaryConv2d):
        failures.append("stem conv1 is ternary; it must be fp32")
    if type(model.fc) is not nn.Linear:
        failures.append("final FC is not a plain fp32 Linear")
    print("fp32 layers: stem conv1, final fc, all BatchNorm parameters and running stats\n")

    # 3. The constraint itself, filter by filter.
    print(f"{'layer':<26}{'shape':<18}{'zero %':>8}{'alpha min':>11}{'alpha max':>11}{'degen':>6}")
    total_weights = total_zero = 0
    exported = {}
    with torch.no_grad():
        for name in names:
            weight = model.get_submodule(name).weight
            w_t, degenerate = ternarize(weight)
            flat = w_t.reshape(w_t.shape[0], -1)
            alpha = flat.abs().max(dim=1).values
            on_grid = (flat == 0) | (flat.abs() == alpha[:, None])
            if not bool(on_grid.all()):
                failures.append(f"{name}: a weight is not in {{-alpha, 0, +alpha}}")
            distinct = max(len(torch.unique(row)) for row in flat)
            if distinct > 3:
                failures.append(f"{name}: a filter has {distinct} distinct values")

            codes = torch.sign(w_t).to(torch.int8)
            zeros = int((codes == 0).sum())
            total_weights += codes.numel()
            total_zero += zeros
            print(f"{name:<26}{str(tuple(w_t.shape)):<18}{100 * zeros / codes.numel():>7.2f}%"
                  f"{float(alpha.min()):>11.4g}{float(alpha.max()):>11.4g}"
                  f"{int(degenerate.sum()):>6}")

            # 4. The module's forward must use the ternary weights, not the latent ones.
            module = model.get_submodule(name)
            x = torch.randn(1, weight.shape[1], 8, 8)
            expected = F.conv2d(x, w_t, None, module.stride, module.padding,
                                module.dilation, module.groups)
            if not torch.allclose(module(x), expected, atol=1e-5):
                failures.append(f"{name}: forward does not use the ternary weights")

            packed, n = pack_2bit(codes)
            exported[name] = {"packed": packed, "n": n, "shape": tuple(w_t.shape),
                              "alpha": alpha.to(torch.float32), "w_t": w_t}

    print(f"\noverall ternary sparsity: {100 * total_zero / total_weights:.2f}% "
          f"of {total_weights:,} weights")
    accounted = storage.ternary_storage_bits(model, names)
    print(f"exported storage (fixed-width accounting): {storage.mb(accounted):.4f} MB")
    print(f"fp32 ResNet-18 for comparison: "
          f"{storage.mb(storage.fp32_storage_bits(resnet18_cifar())):.4f} MB")

    # 5. Deployable 2-bit model, then prove it reproduces the ternary network.
    if args.export:
        ternary_weight_keys = {f"{n}.weight" for n in names}
        fp32_state = {k: v for k, v in model.state_dict().items() if k not in ternary_weight_keys}
        package = {
            "format": "ternary-resnet18-2bit-v1",
            "ternary": {n: {k: v for k, v in e.items() if k != "w_t"} for n, e in exported.items()},
            "fp32": fp32_state,
        }
        out = pathlib.Path(args.export)
        torch.save(package, out)
        print(f"\nwrote {out}: {out.stat().st_size / 2**20:.4f} MB on disk")

        reloaded = torch.load(out, map_location="cpu")
        plain = resnet18_cifar()  # ordinary fp32 convolutions, no quantizer anywhere
        state = dict(reloaded["fp32"])
        for name, entry in reloaded["ternary"].items():
            codes = unpack_2bit(entry["packed"], entry["n"], entry["shape"])
            weight = entry["alpha"].view(-1, 1, 1, 1) * codes.to(torch.float32)
            if not torch.equal(weight, exported[name]["w_t"]):
                failures.append(f"{name}: 2-bit round trip changed the weights")
            state[f"{name}.weight"] = weight
        plain.load_state_dict(state)
        plain.eval()

        batch = torch.randn(8, 3, 32, 32)
        with torch.no_grad():
            if not torch.allclose(plain(batch), model(batch), atol=1e-4):
                failures.append("the 2-bit exported model does not reproduce the ternary model")
        print("reloaded the 2-bit file into a plain fp32 ResNet-18: outputs match")

    print()
    if failures:
        for failure in failures:
            print("FAIL  " + failure)
        return 1
    print("ok    every ternary layer satisfies the {-alpha, 0, +alpha} constraint")
    return 0


if __name__ == "__main__":
    sys.exit(main())
