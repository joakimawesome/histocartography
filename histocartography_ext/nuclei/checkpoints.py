from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

_ENCODE_GROUP_RE = re.compile(r"^encode\.group(?P<g>\d+)\.(?P<rest>.+)$")
_DECODE_RE = re.compile(r"^(?P<pfx>decode_np|decode_hv)\.(?P<rest>.+)$")
_HOVERNET_CLASS = None


def _get_hovernet_class():
    """Load HoverNet without importing histocartography_ext.ml (which may require DGL)."""
    global _HOVERNET_CLASS
    if _HOVERNET_CLASS is not None:
        return _HOVERNET_CLASS

    hovernet_path = (
        Path(__file__).resolve().parents[1] / "ml" / "models" / "hovernet.py"
    )
    if not hovernet_path.is_file():
        raise FileNotFoundError(f"HoverNet implementation not found at: {hovernet_path}")

    spec = importlib.util.spec_from_file_location(
        "_histocartography_ext_hovernet", str(hovernet_path)
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for: {hovernet_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    hovernet_class = getattr(module, "HoverNet", None)
    if hovernet_class is None:
        raise ImportError(f"No HoverNet class found in: {hovernet_path}")

    _HOVERNET_CLASS = hovernet_class
    return hovernet_class


def _looks_like_legacy_desc(d: Mapping[str, Any]) -> bool:
    # Legacy weights shipped in this repo's checkpoints/hovernet_*.pth are in a
    # TF-ish naming format, typically nested under a top-level "desc" key.
    try:
        keys = list(d.keys())
    except Exception:
        return False

    return (
        any(isinstance(k, str) and k.startswith("decoder.") for k in keys)
        and any(isinstance(k, str) and k.startswith("d0.") for k in keys)
        and any(isinstance(k, str) and k.startswith("conv0.") for k in keys)
    )


def _legacy_desc_key_for_state_key(state_key: str) -> Optional[str]:
    # Encoder conv0 / conv_bot
    if state_key == "encode.conv0.conv.weight":
        return "conv0./.weight"
    if state_key.startswith("encode.conv0.act.bn."):
        suf = state_key.split("encode.conv0.act.bn.", 1)[1]
        return f"conv0.bn.{suf}"
    if state_key == "encode.conv_bot.conv.weight":
        return "conv_bot.weight"

    # Residual groups (d0..d3)
    m = _ENCODE_GROUP_RE.match(state_key)
    if m:
        group_id = m.group("g")
        rest = m.group("rest")
        if rest == "block0_convshortcut.conv.weight":
            return f"d{group_id}.shortcut.weight"
        if rest.startswith("bnlast.bn."):
            suf = rest.split("bnlast.bn.", 1)[1]
            return f"d{group_id}.blk_bna.bn.{suf}"

        mm = re.match(r"^block(?P<i>\d+)_preact\.bn\.(?P<suf>.+)$", rest)
        if mm:
            idx = mm.group("i")
            suf = mm.group("suf")
            return f"d{group_id}.units.{idx}.preact/bn.{suf}"

        mm = re.match(r"^block(?P<i>\d+)_conv1\.conv\.weight$", rest)
        if mm:
            idx = mm.group("i")
            return f"d{group_id}.units.{idx}.conv1.weight"
        mm = re.match(r"^block(?P<i>\d+)_conv1\.act\.bn\.(?P<suf>.+)$", rest)
        if mm:
            idx = mm.group("i")
            suf = mm.group("suf")
            return f"d{group_id}.units.{idx}.conv1/bn.{suf}"

        mm = re.match(r"^block(?P<i>\d+)_conv2\.conv\.weight$", rest)
        if mm:
            idx = mm.group("i")
            return f"d{group_id}.units.{idx}.conv2.weight"
        mm = re.match(r"^block(?P<i>\d+)_conv2\.act\.bn\.(?P<suf>.+)$", rest)
        if mm:
            idx = mm.group("i")
            suf = mm.group("suf")
            return f"d{group_id}.units.{idx}.conv2/bn.{suf}"

        mm = re.match(r"^block(?P<i>\d+)_conv3\.conv\.weight$", rest)
        if mm:
            idx = mm.group("i")
            return f"d{group_id}.units.{idx}.conv3.weight"

        return None

    # Decoders (decoder.np / decoder.hv)
    m = _DECODE_RE.match(state_key)
    if m:
        pfx = m.group("pfx")
        rest = m.group("rest")
        branch = "np" if pfx == "decode_np" else "hv"

        mm = re.match(r"^u(?P<u>[23])_conva\.conv\.weight$", rest)
        if mm:
            u_idx = mm.group("u")
            return f"decoder.{branch}.u{u_idx}.conva.weight"

        mm = re.match(r"^u(?P<u>[23])_convf\.conv\.weight$", rest)
        if mm:
            u_idx = mm.group("u")
            return f"decoder.{branch}.u{u_idx}.convf.weight"

        if rest == "u1_conva.conv.weight":
            return f"decoder.{branch}.u1.conva.weight"

        mm = re.match(r"^u(?P<u>[23])_dense\.blk_bna\.bn\.(?P<suf>.+)$", rest)
        if mm:
            u_idx = mm.group("u")
            suf = mm.group("suf")
            return f"decoder.{branch}.u{u_idx}.dense.blk_bna.bn.{suf}"

        mm = re.match(
            r"^u(?P<u>[23])_dense\.blk_(?P<i>\d+)preact_bna\.bn\.(?P<suf>.+)$",
            rest,
        )
        if mm:
            u_idx = mm.group("u")
            idx = mm.group("i")
            suf = mm.group("suf")
            return f"decoder.{branch}.u{u_idx}.dense.units.{idx}.preact_bna/bn.{suf}"

        mm = re.match(r"^u(?P<u>[23])_dense\.blk_(?P<i>\d+)conv1\.conv\.weight$", rest)
        if mm:
            u_idx = mm.group("u")
            idx = mm.group("i")
            return f"decoder.{branch}.u{u_idx}.dense.units.{idx}.conv1.weight"
        mm = re.match(
            r"^u(?P<u>[23])_dense\.blk_(?P<i>\d+)conv1\.act\.bn\.(?P<suf>.+)$",
            rest,
        )
        if mm:
            u_idx = mm.group("u")
            idx = mm.group("i")
            suf = mm.group("suf")
            return f"decoder.{branch}.u{u_idx}.dense.units.{idx}.conv1/bn.{suf}"

        mm = re.match(r"^u(?P<u>[23])_dense\.blk_(?P<i>\d+)conv2\.conv\.weight$", rest)
        if mm:
            u_idx = mm.group("u")
            idx = mm.group("i")
            return f"decoder.{branch}.u{u_idx}.dense.units.{idx}.conv2.weight"

        return None

    # Output heads (u0)
    mm = re.match(r"^preact_out_(?P<br>np|hv)\.bn\.(?P<suf>.+)$", state_key)
    if mm:
        branch = mm.group("br")
        suf = mm.group("suf")
        return f"decoder.{branch}.u0.bn.{suf}"

    mm = re.match(r"^conv_out_(?P<br>np|hv)\.conv\.(?P<suf>weight|bias)$", state_key)
    if mm:
        branch = mm.group("br")
        suf = mm.group("suf")
        return f"decoder.{branch}.u0.conv.{suf}"

    return None


def _build_hovernet_from_legacy_desc(
    desc: Mapping[str, torch.Tensor],
) -> torch.nn.Module:
    hovernet_class = _get_hovernet_class()
    model = hovernet_class()

    state_dict: dict[str, torch.Tensor] = {}
    missing: list[str] = []
    for state_key in model.state_dict().keys():
        # Some torch versions treat BatchNorm's num_batches_tracked as optional; we
        # never store it in legacy checkpoints.
        if state_key.endswith("num_batches_tracked"):
            continue

        desc_key = _legacy_desc_key_for_state_key(state_key)
        if desc_key is None:
            missing.append(state_key)
            continue

        try:
            state_dict[state_key] = desc[desc_key]
        except KeyError:
            missing.append(state_key)

    if missing:
        example = ", ".join(missing[:10])
        raise KeyError(
            f"Legacy HoVer-Net checkpoint is missing {len(missing)} required keys "
            f"(e.g. {example})."
        )

    incompatible = model.load_state_dict(state_dict, strict=False)
    missing_nontracked = [
        k for k in incompatible.missing_keys if not k.endswith("num_batches_tracked")
    ]
    if missing_nontracked:
        example = ", ".join(missing_nontracked[:10])
        raise RuntimeError(f"Failed to load HoVer-Net weights; missing keys: {example}")

    return model


def _extract_state_dict(obj: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    for key in ("state_dict", "model_state_dict", "model", "net", "network"):
        val = obj.get(key)
        if isinstance(val, Mapping):
            return val
    # Sometimes the dict itself is a state_dict (string -> tensor mapping).
    if obj and all(isinstance(k, str) for k in obj.keys()):
        n_tensors = sum(1 for v in obj.values() if isinstance(v, torch.Tensor))
        if n_tensors >= max(1, int(0.8 * len(obj))):
            return obj
    return None


def _strip_prefix(state_dict: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    if not prefix:
        return dict(state_dict)
    if not any(isinstance(k, str) and k.startswith(prefix) for k in state_dict.keys()):
        return dict(state_dict)
    out: dict[str, Any] = {}
    for k, v in state_dict.items():
        if isinstance(k, str) and k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[str(k)] = v
    return out


def _build_hovernet_from_state_dict(
    state_dict: Mapping[str, Any],
) -> torch.nn.Module:
    hovernet_class = _get_hovernet_class()
    model = hovernet_class()

    target_keys = set(model.state_dict().keys())
    prefix_sets = [
        [],
        ["module."],
        ["model."],
        ["net."],
        ["hovernet."],
        ["module.", "model."],
        ["module.", "net."],
        ["module.", "hovernet."],
    ]

    best_sd: Mapping[str, Any] = state_dict
    best_score = -1
    for prefixes in prefix_sets:
        sd = dict(state_dict)
        for pfx in prefixes:
            sd = _strip_prefix(sd, pfx)
        score = sum(1 for k in sd.keys() if k in target_keys)
        if score > best_score:
            best_score = score
            best_sd = sd

    incompatible = model.load_state_dict(best_sd, strict=False)
    missing_nontracked = [
        k for k in incompatible.missing_keys if not k.endswith("num_batches_tracked")
    ]
    if missing_nontracked:
        example = ", ".join(missing_nontracked[:10])
        raise RuntimeError(
            f"State-dict checkpoint is missing {len(missing_nontracked)} HoVer-Net keys "
            f"(e.g. {example})."
        )

    return model


def load_hovernet_checkpoint(
    path: str | Path,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """
    Load a HoVer-Net checkpoint.

    Supported formats:
      - Full model object (torch.save(model, path))
      - State dict mapping (torch.save(model.state_dict(), path))
      - Legacy 'desc' mapping in checkpoints/hovernet_*.pth
    """
    _ = device  # device is applied by the caller via model.to(...)
    obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, torch.nn.Module):
        return obj

    if isinstance(obj, Mapping):
        desc = None
        try:
            maybe_desc = obj["desc"]
            if isinstance(maybe_desc, Mapping):
                desc = maybe_desc
        except Exception:
            desc = None

        if desc is not None and _looks_like_legacy_desc(desc):
            return _build_hovernet_from_legacy_desc(desc)  # type: ignore[arg-type]
        if _looks_like_legacy_desc(obj):
            return _build_hovernet_from_legacy_desc(obj)  # type: ignore[arg-type]

        state_dict = _extract_state_dict(obj)
        if state_dict is not None:
            return _build_hovernet_from_state_dict(state_dict)

    raise TypeError(
        f"Unsupported HoVer-Net checkpoint at {path}: got {type(obj).__name__}. "
        "Expected a torch.nn.Module, a state_dict-like mapping, or a legacy 'desc' checkpoint."
    )


__all__ = [
    "load_hovernet_checkpoint",
]
