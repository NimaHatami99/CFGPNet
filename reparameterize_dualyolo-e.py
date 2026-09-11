#!/usr/bin/env python3
r"""Convert dualyolo2-e training weights to a compact dualgelan2-e checkpoint.

Put this script in the SAME custom YOLO repository used for training. Its
models.yolo.Model and custom Conv, DualELA and FeatFuse implementations are
required. Use the original, unfused training checkpoint.

Run from that repository's root:
    python reparameterize_dualyolo2_e_compact.py --weights best.pt \
        --cfg models/detect/dualgelan2-e.yaml \
        --output best_dualgelan2-e-fp16.pt --verify

FP16 storage is the DEFAULT, matching the supplied model.half() example.
Use --fp32 to retain FP32 storage. For 129 million parameters, parameter
storage is approximately 258 MB in FP16 or 516 MB in FP32, plus metadata and
buffers. Storage precision does not determine inference execution precision.

Destination <- trained source:
    model.0..56.*   <- model.0..56.*
    model.57..75.*  <- model.70..88.*
    model.76.cv2.*  <- model.89.cv4.*
    model.76.cv3.*  <- model.89.cv5.*
    model.76.dfl.*  <- model.89.dfl2.*

The default source is checkpoint['model'], as in the supplied example. Use
--ema if the original evaluation used checkpoint['ema']. Standard YOLOv9
attempt_load prefers EMA when present; do not compare different weight sets.

--verify checks raw and decoded MAIN-branch outputs on a reproducible
six-channel input, before conversion to storage precision and after loading
the saved file. Both comparisons execute in FP32 on CPU. For the second
comparison, the source receives the same storage-precision conversion.
Any prediction drift from FP16 rounding is reported separately. These checks
are not dataset evaluation or a guarantee of identical mAP. Re-evaluate with
the same data, preprocessing, NMS, image size and execution precision, using
the DDetect/single-head validation path for the converted checkpoint.
"""

import argparse
from collections import Counter
from copy import deepcopy
import inspect
import os
from pathlib import Path
import tempfile


def source_key_for(target_key):
    """Map an entire state key, including its unambiguous layer number."""
    parts = target_key.split(".", 2)
    if len(parts) != 3 or parts[0] != "model" or not parts[1].isdigit():
        raise ValueError(f"Unexpected destination state key: {target_key}")
    index, suffix = int(parts[1]), parts[2]
    if index < 57:
        return target_key
    if index < 76:
        return f"model.{index + 13}.{suffix}"
    if index == 76:
        branch, separator, rest = suffix.partition(".")
        branches = {"cv2": "cv4", "cv3": "cv5", "dfl": "dfl2"}
        if branch in branches and separator:
            return f"model.89.{branches[branch]}.{rest}"
    raise ValueError(f"No conversion rule for destination state key: {target_key}")


def validate_architectures(source_cfg, target_cfg):
    """Check both layer definitions and graph connections in the supplied YAMLs."""
    source = source_cfg["backbone"] + source_cfg["head"]
    target = target_cfg["backbone"] + target_cfg["head"]
    if len(source) != 90 or len(target) != 77:
        raise ValueError("Expected 90 trained layers and 77 inference layers.")
    if source[-1][2] != "DualDDetect" or target[-1][2] != "DDetect":
        raise ValueError("Expected DualDDetect at 89 and DDetect at 76.")
    for key, default in (("ch", 6), ("depth_multiple", 1.0),
                         ("width_multiple", 1.0), ("activation", None)):
        if source_cfg.get(key, default) != target_cfg.get(key, default):
            raise ValueError(f"Source and destination differ in {key}.")
    if target_cfg.get("ch", 6) != 6:
        raise ValueError("These architectures require six input channels.")

    layer_map = {i: i if i < 57 else i + 13 for i in range(76)}

    def inputs(layer, index):
        refs = layer[0] if isinstance(layer[0], list) else [layer[0]]
        return [r if r >= 0 else index + r for r in refs]

    for target_index, source_index in layer_map.items():
        if target[target_index][1:] != source[source_index][1:]:
            raise ValueError(
                f"Layer definition mismatch: target {target_index}, source {source_index}."
            )
        expected_inputs = [
            layer_map[r] if r >= 0 else r
            for r in inputs(target[target_index], target_index)
        ]
        if expected_inputs != inputs(source[source_index], source_index):
            raise ValueError(f"Input connection mismatch at target {target_index}.")

    if source[-1][0] != [69, 66, 63, 82, 85, 88]:
        raise ValueError("Unexpected auxiliary/main detection inputs in source YAML.")
    if target[-1][0] != [69, 72, 75]:
        raise ValueError("Unexpected detection inputs in inference YAML.")
    if [layer_map[r] for r in target[-1][0]] != source[-1][0][3:]:
        raise ValueError("Inference detection inputs must select the main branch.")


def remap_state_dict(source_state, target_state):
    """Require a trained tensor with the correct shape for EVERY target entry."""
    converted = {}
    for target_key, target_tensor in target_state.items():
        source_key = source_key_for(target_key)
        if source_key not in source_state:
            raise KeyError(
                f"Missing {source_key} -> {target_key}. "
                "Use an original, unfused dualyolo2-e checkpoint."
            )
        source_tensor = source_state[source_key]
        if source_tensor.shape != target_tensor.shape:
            raise ValueError(
                f"Shape mismatch: {source_key} {tuple(source_tensor.shape)} -> "
                f"{target_key} {tuple(target_tensor.shape)}. "
                "Check the YAML and use the custom model code from training."
            )
        converted[target_key] = source_tensor
    return converted


def load_full_checkpoint(path):
    """Load the user's trusted full-model YOLO checkpoint, including on torch 2.6+."""
    import torch

    options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = False
    return torch.load(str(path), **options)


def reset_detection_metadata(model):
    """Restore P3/P4/P5 decoding metadata after a dtype change or forward pass."""
    head = model.model[-1]
    if head.nl != 3:
        raise ValueError(f"Expected three detection scales, found {head.nl}.")
    reference = next(model.parameters())
    head.stride = reference.new_tensor([8.0, 16.0, 32.0])
    model.stride = head.stride
    head.shape = None
    head.anchors = reference.new_empty(0)
    head.strides = reference.new_empty(0)
    head.export = False


def verify_main_branch(source, target, image_size, label):
    """Check standard DualDDetect's second branch against DDetect, before NMS."""
    import torch

    source.float().eval()
    target.float().eval()
    source_head = source.model[-1]
    source_head.export = False
    source_head.shape = None
    reset_detection_metadata(target)
    torch.testing.assert_close(
        source_head.stride.cpu(), target.model[-1].stride.cpu(), rtol=0, atol=0
    )
    images = torch.rand(
        1, 6, image_size, image_size,
        generator=torch.Generator().manual_seed(0),
    )
    with torch.inference_mode():
        source_output = source(images.clone())
        target_output = target(images.clone())

    # DualDDetect: ([aux_pred, main_pred], [aux_raw, main_raw]).
    # DDetect: (pred, raw). Never pass the auxiliary predictions to NMS.
    if not (
        isinstance(source_output, tuple) and len(source_output) == 2
        and isinstance(source_output[0], (list, tuple))
        and len(source_output[0]) == 2
        and isinstance(source_output[1], (list, tuple))
        and len(source_output[1]) == 2
        and isinstance(target_output, tuple) and len(target_output) == 2
    ):
        raise RuntimeError("Verification expects standard DualDDetect/DDetect outputs.")
    main_prediction, main_raw = source_output[0][1], source_output[1][1]
    prediction, raw = target_output
    if len(main_raw) != 3 or len(raw) != 3:
        raise RuntimeError("Expected three raw output scales in both models.")
    for actual, expected in [(prediction, main_prediction), *zip(raw, main_raw)]:
        if not (torch.isfinite(actual).all() and torch.isfinite(expected).all()):
            raise RuntimeError("Non-finite predictions encountered during verification.")
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    error = (prediction - main_prediction).abs().max().item()
    print(f"{label}: PASSED; max main-prediction difference = {error:.6g}")
    return prediction.detach().cpu()


def check_storage_precision(model, dtype):
    """Check for invalid values/overflow and report any added FP16 rounding."""
    import torch

    rounded_tensors = 0
    for key, tensor in model.state_dict().items():
        if not tensor.is_floating_point():
            continue
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite trained parameter/buffer: {key}")
        stored = tensor.to(dtype=dtype)
        if not torch.isfinite(stored).all():
            raise ValueError(f"FP16 overflow in {key}; use --fp32 for this checkpoint.")
        if not torch.equal(stored.to(dtype=tensor.dtype), tensor):
            rounded_tensors += 1
    if rounded_tensors:
        print(
            f"FP16 rounding changes values in {rounded_tensors} retained state tensors. "
            "Re-evaluate mAP; use --fp32 to avoid this rounding."
        )
    else:
        print(f"All retained floating state values are exact in {dtype}; no added rounding.")
    return rounded_tensors


def verify_saved_state(expected_model, loaded_model, dtype):
    """Check disk round-trip values and dtype before running any FP32 forwards."""
    import torch

    expected = expected_model.state_dict()
    actual = loaded_model.state_dict()
    if expected.keys() != actual.keys():
        raise RuntimeError("Saved checkpoint state keys changed during serialization.")
    for key, tensor in actual.items():
        if tensor.dtype != expected[key].dtype or not torch.equal(tensor, expected[key]):
            raise RuntimeError(f"Saved checkpoint differs at {key}.")
        if tensor.is_floating_point() and tensor.dtype != dtype:
            raise RuntimeError(f"Unexpected saved dtype at {key}: {tensor.dtype}.")
    print("Saved checkpoint: exact tensor round-trip and storage dtype verified.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--weights", type=Path, default=Path("best.pt"))
    parser.add_argument("--cfg", type=Path, default=Path("models/detect/dualgelan2-e.yaml"))
    parser.add_argument("--output", type=Path,
                        help="Default: <weights-stem>_dualgelan2-e-fp16.pt (or fp32).")
    parser.add_argument("--ema", action="store_true", help="Select checkpoint['ema'].")
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("--fp32", dest="half", action="store_false",
                           help="Save FP32; approximately twice the FP16 file size.")
    precision.add_argument("--half", dest="half", action="store_true",
                           help="Save FP16 (already the default).")
    parser.set_defaults(half=True)
    parser.add_argument("--verify", action="store_true",
                        help="Compare main predictions before export and after reload.")
    parser.add_argument("--imgsz", type=int, default=256,
                        help="Verification input size: a positive multiple of 32.")
    args = parser.parse_args()
    precision_name = "fp16" if args.half else "fp32"
    if args.output is None:
        args.output = args.weights.with_name(
            f"{args.weights.stem}_dualgelan2-e-{precision_name}.pt"
        )
    if not args.weights.is_file():
        parser.error(f"Checkpoint not found: {args.weights}")
    if not args.cfg.is_file():
        parser.error(f"Inference YAML not found: {args.cfg}")
    if args.weights.resolve() == args.output.resolve():
        parser.error("Output must differ from the original checkpoint.")
    if args.imgsz < 32 or args.imgsz % 32:
        parser.error("--imgsz must be a positive multiple of 32.")

    import torch
    import yaml
    from models.yolo import Model

    checkpoint = load_full_checkpoint(args.weights)
    source_name = "ema" if args.ema else "model"
    if not isinstance(checkpoint, dict) or checkpoint.get(source_name) is None:
        raise ValueError(f"Checkpoint must contain a '{source_name}' model.")
    trained = checkpoint[source_name]
    if isinstance(trained, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        trained = trained.module
    if not isinstance(trained, torch.nn.Module) or not isinstance(getattr(trained, "yaml", None), dict):
        raise TypeError("Expected a full YOLO model with its saved YAML.")
    if len(trained.model) != 90 or type(trained.model[-1]).__name__ != "DualDDetect":
        raise ValueError("Expected dualyolo2-e with DualDDetect at layer 89.")
    print(f"Using checkpoint['{source_name}'].")
    if not args.ema and checkpoint.get("ema") is not None:
        print("EMA is also present: use --ema if the original evaluation used EMA.")
    del checkpoint  # Release optimizer/other model copies; never put them in the output.
    trained.cpu().eval()
    source_dtypes = Counter(str(p.dtype) for p in trained.parameters())
    print(f"Source parameter tensor dtypes: {dict(source_dtypes)}")
    for key, tensor in trained.state_dict().items():
        if tensor.is_complex() or (
            tensor.is_floating_point()
            and tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        ):
            raise TypeError(f"Unsupported source dtype at {key}: {tensor.dtype}.")

    with args.cfg.open(encoding="utf-8") as stream:
        target_cfg = yaml.safe_load(stream)
    validate_architectures(trained.yaml, target_cfg)
    expected_strides = torch.tensor([8.0, 16.0, 32.0])
    torch.testing.assert_close(
        trained.model[-1].stride.detach().cpu().float(), expected_strides, rtol=0, atol=0
    )

    # nc must be supplied at construction, before loading classifier tensors.
    nc = int(trained.model[-1].nc)
    model = Model(str(args.cfg), ch=6, nc=nc, anchors=3).cpu().float().eval()
    validate_architectures(trained.yaml, model.yaml)
    if len(model.model) != 77 or type(model.model[-1]).__name__ != "DDetect":
        raise ValueError("Expected dualgelan2-e with DDetect at layer 76.")
    converted = remap_state_dict(trained.state_dict(), model.state_dict())
    model.load_state_dict(converted, strict=True)  # Copies into the fresh target model.
    print(f"Copied all {len(converted)} target state entries; nc={nc}, ch=6.")
    del converted
    model.names = deepcopy(getattr(trained, "names", [str(i) for i in range(nc)]))
    model.nc = nc
    model.yaml["nc"] = nc
    reset_detection_metadata(model)
    source_parameters = sum(p.numel() for p in trained.parameters())
    target_parameters = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {source_parameters:,} -> {target_parameters:,}.")

    original_prediction = None
    if args.verify:
        original_prediction = verify_main_branch(
            trained, model, args.imgsz, "Original weights / FP32 execution"
        )

    storage_dtype = torch.float16 if args.half else torch.float32
    check_storage_precision(model, storage_dtype)
    model.half() if args.half else model.float()
    model.eval().requires_grad_(False)
    reset_detection_metadata(model)
    # Never call bias_init() after loading the trained detection biases.
    payload = {
        "model": model,
        "optimizer": None, "best_fitness": None, "ema": None,
        "updates": None, "opt": None, "git": None, "date": None, "epoch": -1,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # A failed verification must not publish or replace the requested output.
    with tempfile.NamedTemporaryFile(
        dir=args.output.parent, prefix=args.output.stem + ".", suffix=".pt", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(payload, str(temporary_path))
        if args.verify:
            loaded = load_full_checkpoint(temporary_path)["model"]
            verify_saved_state(model, loaded, storage_dtype)
            if args.half:
                # Compare like precision: the retained main weights get exactly
                # the same cast as the saved model. The original stays on disk.
                trained.half().float()
            saved_prediction = verify_main_branch(
                trained, loaded, args.imgsz,
                f"Reloaded {precision_name.upper()} weights / FP32 execution",
            )
            drift = (saved_prediction - original_prediction).abs().max().item()
            print(f"Original-to-export prediction drift from storage cast: max abs = {drift:.6g}")
            print("Prediction checks passed; dataset mAP and latency were not measured.")
        os.replace(temporary_path, args.output)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    size = args.output.stat().st_size
    source_size = args.weights.stat().st_size
    print(f"Saved {args.output}: {size / 1e6:.2f} MB ({size / 2**20:.2f} MiB), {precision_name.upper()}.")
    print(f"Original checkpoint: {source_size / 1e6:.2f} MB.")
    if not args.verify:
        print("Prediction verification was not run; add --verify to run it.")


if __name__ == "__main__":
    main()
