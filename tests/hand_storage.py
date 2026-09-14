"""Independent by-hand storage recomputation (prereg §9). No torch: pure arithmetic,
so it is a genuine cross-check of src/storage.py rather than the same code twice."""
FP32, TERN = 32, 2
MB = 8 * 1024 * 1024


def r18_like(blocks, chans, option_a):
    """Returns (conv_weights_outside_stem, n_out_filters_outside_stem, stem, bn_elems, fc)."""
    stem = 3 * chans[0] * 9
    bn_elems = chans[0]            # one BN after the stem (counted per-tensor below)
    convs, filters = 0, 0
    inp = chans[0]
    for i, (nb, ch) in enumerate(zip(blocks, chans)):
        for b in range(nb):
            convs += inp * ch * 9 + ch * ch * 9     # conv1 (stride only changes spatial), conv2
            filters += ch + ch
            bn_elems += ch + ch                      # bn1, bn2
            if inp != ch or (i > 0 and b == 0):
                if not option_a:
                    convs += inp * ch * 1            # 1x1 downsample conv
                    filters += ch
                    bn_elems += ch                   # downsample BN
            inp = ch
    fc = chans[-1] * 10 + 10
    return convs, filters, stem, bn_elems, fc


def fp32_bits(blocks, chans, option_a):
    convs, _, stem, bn_elems, fc = r18_like(blocks, chans, option_a)
    params = convs + stem + fc + 2 * bn_elems        # gamma + beta are parameters
    buffers = 2 * bn_elems                           # running_mean + running_var
    return (params + buffers) * FP32


def fp32_bits_preregliteral(blocks, chans, option_a):
    """prereg §6 as literally written: param_count*32 + BN params AND buffers again."""
    convs, _, stem, bn_elems, fc = r18_like(blocks, chans, option_a)
    params = convs + stem + fc + 2 * bn_elems
    return params * FP32 + (2 * bn_elems + 2 * bn_elems) * FP32


def n_params(blocks, chans, option_a):
    convs, _, stem, bn_elems, fc = r18_like(blocks, chans, option_a)
    return convs + stem + fc + 2 * bn_elems


def ternary_bits(blocks, chans, option_a):
    convs, filters, stem, bn_elems, fc = r18_like(blocks, chans, option_a)
    return (convs * TERN + filters * FP32) + (stem + fc + 4 * bn_elems) * FP32


def report():
    R18 = ([2, 2, 2, 2], [64, 128, 256, 512], False)
    target = ternary_bits(*R18)
    print(f"ternary R18 params={n_params(*R18):,}  exported={target:,} bits = {target/MB:.4f} MB")
    print(f"fp32    R18 exported={fp32_bits(*R18)/MB:.4f} MB\n")

    print(f"{'depth':>6} {'params':>10} {'MB (correct)':>13} {'|d|/target':>11} "
          f"{'MB (literal)':>13} {'|d|/target':>11}   optB params")
    rows = []
    for depth, n in {20: 3, 32: 5, 44: 7, 56: 9, 110: 18}.items():
        blocks, chans = [n] * 3, [16, 32, 64]
        ok, lit = fp32_bits(blocks, chans, True), fp32_bits_preregliteral(blocks, chans, True)
        p_a, p_b = n_params(blocks, chans, True), n_params(blocks, chans, False)
        rows.append((depth, abs(ok - target) / target, abs(lit - target) / target))
        print(f"{depth:>6} {p_a:>10,} {ok/MB:>13.4f} {abs(ok-target)/target:>11.4f} "
              f"{lit/MB:>13.4f} {abs(lit-target)/target:>11.4f}   {p_b:>10,}")

    best_ok = min(rows, key=lambda r: r[1])
    best_lit = min(rows, key=lambda r: r[2])
    print(f"\nargmin correct convention : ResNet-{best_ok[0]}")
    print(f"argmin prereg-literal     : ResNet-{best_lit[0]}")
    print(f"5% tie-break band members : {[r[0] for r in rows if abs(r[1]-best_ok[1]) < 0.05]}")


if __name__ == "__main__":
    report()
