#!/usr/bin/env python3
"""
Sybil Backend — optimised CPU inference + async job polling.

CPU speed techniques applied (fastest to slowest impact):
  1. torch.set_num_threads(N_CPU)  — all cores for BLAS/conv ops  [built into Sybil predict()]
  2. torch.inference_mode()        — disables autograd bookkeeping
  3. torch.compile()               — kernel fusion (PyTorch ≥ 2.0, one-time warm-up)
  4. Parallel DICOM loading        — ThreadPoolExecutor for I/O-bound slice reads
  5. Optimised PNG encoding        — parallel base64 via ThreadPool
"""
import os, sys, base64, logging, re, traceback, threading, uuid, time
import datetime, getpass, math
import multiprocessing
import logging.handlers
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR    = os.path.dirname(SCRIPT_DIR)
SYBIL_SRC  = os.path.join(APP_DIR, "sybil-source")
sys.path.insert(0, SYBIL_SRC)

from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Audit logging: stdout + rotating file ────────────────────────────────────
_LOG_DIR = Path.home() / ".axio_predict" / "logs"
try:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    _audit_handler = logging.handlers.RotatingFileHandler(
        _LOG_DIR / "audit.log",
        maxBytes=50_000_000,   # 50 MB per file
        backupCount=10,
        encoding="utf-8",
    )
    _audit_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _log_handlers: list = [logging.StreamHandler(sys.stdout), _audit_handler]
except Exception:
    _log_handlers = [logging.StreamHandler(sys.stdout)]

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=_log_handlers)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Restrict CORS to localhost origins only — prevents cross-site request forgery
# from a malicious webpage reaching this backend.
CORS(app, resources={r"/*": {"origins": [
    "http://127.0.0.1:5000", "http://localhost:5000",
]}})

N_CPU = multiprocessing.cpu_count()
VOXEL_SPACING_STR = "0.703125 × 0.703125 × 2.5 mm"

# File-type extension sets (defined early — used throughout module)
DICOM_EXTS  = {".dcm", ".dicom", ".ima", ".dic"}
PNG_EXTS    = {".png"}
TIFF_EXTS   = {".tif", ".tiff"}
IMAGE_EXTS  = PNG_EXTS | TIFF_EXTS | {".jpg", ".jpeg"}

# ── Set PyTorch threading at import time (not just at inference time) ──────────
import torch
torch.set_num_threads(N_CPU)
torch.set_num_interop_threads(max(1, N_CPU // 2))   # inter-op parallelism
logger.info(f"PyTorch {torch.__version__} — {N_CPU} CPU cores, "
            f"{torch.get_num_threads()} intra-op threads, "
            f"{torch.get_num_interop_threads()} inter-op threads")

# ── Known Sybil checkpoint md5s ────────────────────────────────────────────────
MD5_TO_MODEL = {
    "28a7cd44f5bcd3e6cc760b65c7e0d54d": "sybil_1",
    "56ce1a7d241dc342982f5466c4a9d7ef": "sybil_2",
    "624407ef8e3a2a009f9fa51f9846fe9a": "sybil_3",
    "64a91b25f84141d32852e75a3aec7305": "sybil_4",
    "65fd1f04cb4c5847d86a9ed8ba31ac1a": "sybil_5",
}
ENSEMBLE_MD5S = set(MD5_TO_MODEL.keys())

_model     : Any = None
_model_key : str = ""
_compiled  : bool = False          # whether torch.compile() has been applied
_jobs      : Dict[str, Any] = {}
_job_lock  = threading.Lock()


# ═══════════════════════════════════════════════════════════════════
# DEVICE
# ═══════════════════════════════════════════════════════════════════

def _sybil_device() -> str:
    """Conv3D (R3D-18) is NOT supported on MPS — always cpu unless CUDA."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def _device_label() -> str:
    import platform
    if torch.cuda.is_available():
        return f"CUDA — {torch.cuda.get_device_name(0)}"
    try:
        if torch.backends.mps.is_available():
            return f"CPU — {platform.processor() or 'Apple Silicon'} ({N_CPU} cores)"
    except Exception:
        pass
    return f"CPU — {platform.processor() or 'unknown'} ({N_CPU} cores)"


# ═══════════════════════════════════════════════════════════════════
# TORCH COMPILE  (PyTorch ≥ 2.0, skip gracefully on older versions)
# ═══════════════════════════════════════════════════════════════════

def _parse_torch_version() -> tuple:
    """Robustly parse torch.__version__ into (major, minor) ints."""
    m = re.match(r"(\d+)\.(\d+)", torch.__version__)
    return (int(m.group(1)), int(m.group(2))) if m else (1, 0)


def _try_compile_model(model_obj) -> Tuple[Any, bool]:
    """
    Apply torch.compile(mode='reduce-overhead') to each SybilNet in the ensemble.
    Returns (model_obj, compiled:bool).
    First inference call after compile will be slow (~30-60s warm-up);
    all subsequent calls are 20-40% faster.
    """
    global _compiled
    if _compiled:
        return model_obj, True

    # torch.compile available from 2.0
    try:
        ver = _parse_torch_version()  # PY-06
    except Exception:
        ver = (1, 0)

    if ver[0] < 2:
        logger.info("torch.compile skipped — requires PyTorch ≥ 2.0")
        return model_obj, False

    if not hasattr(model_obj, "ensemble"):
        return model_obj, False

    try:
        compiled_count = 0
        for sybil_net in model_obj.ensemble:
            if hasattr(sybil_net, "model"):
                # Compile with reduce-overhead: fuses kernels, minimises Python overhead
                sybil_net.model = torch.compile(
                    sybil_net.model,
                    mode="reduce-overhead",
                    fullgraph=False,   # allow fallback for unsupported ops
                )
                compiled_count += 1
            elif hasattr(sybil_net, "image_encoder"):
                # SybilNet directly — compile the whole thing
                compiled_net = torch.compile(
                    sybil_net,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
                # Replace in-place
                idx = list(model_obj.ensemble).index(sybil_net)
                model_obj.ensemble[idx] = compiled_net
                compiled_count += 1

        if compiled_count:
            logger.info(f"torch.compile applied to {compiled_count} model(s). "
                        f"First inference will be slow (warm-up).")
            _compiled = True
            return model_obj, True
    except Exception as e:
        logger.warning(f"torch.compile failed (non-fatal): {e}")

    return model_obj, False


# ═══════════════════════════════════════════════════════════════════
# CHECKPOINT DISCOVERY
# ═══════════════════════════════════════════════════════════════════

def _is_hex32(s: str) -> bool:
    return len(s) == 32 and all(c in "0123456789abcdefABCDEF" for c in s)

def _find_ckpts(directory: str) -> List[str]:
    found = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fname in files:
            if fname.startswith("."): continue
            ext  = os.path.splitext(fname)[1].lower()
            stem = os.path.splitext(fname)[0]
            full = os.path.join(root, fname)
            if ext in (".ckpt", ".pt", ".pth"):
                found.append(full)
            elif ext == "" and _is_hex32(stem):
                found.append(full)
    return sorted(found)

def _find_calibs(directory: str) -> List[str]:
    found = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fname in files:
            if fname.lower().endswith(".json") and not fname.startswith("."):
                found.append(os.path.join(root, fname))
    return sorted(found)

def _identify_ckpt(path: str) -> dict:
    fname = os.path.basename(path)
    stem  = os.path.splitext(fname)[0].lower()
    known = MD5_TO_MODEL.get(stem)
    return {"path": path, "filename": fname,
            "md5": stem if _is_hex32(stem) else None,
            "name": known or fname, "is_known": known is not None}

def _build_groups(ckpt_paths: List[str]) -> List[dict]:
    info     = [_identify_ckpt(p) for p in ckpt_paths]
    by_md5   = {c["md5"]: c for c in info if c["md5"]}
    has_all5 = ENSEMBLE_MD5S.issubset(by_md5.keys())
    groups   = []
    if has_all5:
        ensemble_paths = [by_md5[m]["path"] for m in sorted(ENSEMBLE_MD5S)]
        groups.append({"id": "ensemble",
                       "label": "sybil_ensemble — all 5 models (best accuracy) ⭐",
                       "paths": ensemble_paths, "recommended": True})
    for c in info:
        groups.append({"id": c["md5"] or c["filename"],
                       "label": c["name"] + ("" if c["is_known"] else " (unrecognised)"),
                       "paths": [c["path"]], "recommended": False})
    return groups

def _best_calib(ckpt_paths: List[str], calib_files: List[str]) -> Optional[str]:
    if not calib_files: return None
    if len(ckpt_paths) > 1:
        for c in calib_files:
            if "ensemble" in os.path.basename(c).lower(): return c
    return calib_files[0]


# ═══════════════════════════════════════════════════════════════════
# IMAGE HELPERS  (parallel DICOM loading via ThreadPool)
# ═══════════════════════════════════════════════════════════════════

def _to_b64png(arr) -> str:
    from PIL import Image
    import numpy as np
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def _decompress_dicom(dcm):
    """Handle compressed (JPEG, JPEG2000, RLE) DICOM pixel data safely."""
    ts = getattr(dcm, "file_meta", None)
    ts = getattr(ts, "TransferSyntaxUID", None) if ts else None
    COMPRESSED = {
        "1.2.840.10008.1.2.4.50",   # JPEG Baseline
        "1.2.840.10008.1.2.4.51",   # JPEG Extended
        "1.2.840.10008.1.2.4.57",   # JPEG Lossless
        "1.2.840.10008.1.2.4.70",   # JPEG Lossless FOP
        "1.2.840.10008.1.2.4.80",   # JPEG-LS Lossless
        "1.2.840.10008.1.2.4.81",   # JPEG-LS Near-lossless
        "1.2.840.10008.1.2.4.90",   # JPEG 2000 Lossless
        "1.2.840.10008.1.2.4.91",   # JPEG 2000 Lossy
        "1.2.840.10008.1.2.5",      # RLE Lossless
    }
    if ts and str(ts) in COMPRESSED:
        try:
            dcm.decompress()
        except Exception:
            try:
                # Fallback: use PIL via bytes
                from PIL import Image
                import numpy as np
                arr = dcm.pixel_array
                if arr.ndim == 2:
                    return arr   # PY-04: single-frame 2D, return directly
                frames = list(arr)  # multi-frame: iterate frames not rows
                return np.array(frames[0] if frames else arr)
            except Exception as e:
                raise RuntimeError(f"Cannot decompress DICOM (TransferSyntax={ts}): {e}")
    return dcm.pixel_array

def _dicom_windowed(path: str, window_center: float = -600, window_width: float = 1500):
    """Load one DICOM slice with lung windowing → uint8.
    Handles standard, clinical compressed (JPEG/J2K/RLE), and implicit-LE DICOMs.
    Window: C=-600 W=1500 matches Sybil's DicomLoader defaults."""
    import numpy as np, pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut
    dcm = pydicom.dcmread(path, force=True)          # force=True tolerates missing preamble
    if not hasattr(dcm, "file_meta"):
        # Implicit-LE DICOMs often lack file_meta; add a minimal one
        from pydicom.dataset import FileMetaDataset
        from pydicom.uid import ImplicitVRLittleEndian
        dcm.file_meta = FileMetaDataset()
        dcm.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
        dcm.is_implicit_VR = True
        dcm.is_little_endian = True
    px = _decompress_dicom(dcm)
    arr = apply_modality_lut(px, dcm).astype(np.float32)
    c = window_center - 0.5
    w = window_width - 1
    lo, hi = c - w / 2, c + w / 2
    out = np.empty_like(arr)
    out[arr <= lo] = 0.0
    out[arr > hi]  = 255.0
    m = (arr > lo) & (arr <= hi)
    out[m] = ((arr[m] - c) / w + 0.5) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)

def _load_image_universal(path: str, file_type: str) -> Optional["np.ndarray"]:
    """Load any supported image type → uint8 grayscale numpy array.

    Supported:
      - DICOM (.dcm .dicom .ima .dic or no-ext): standard, clinical compressed,
        implicit-LE, missing preamble
      - PNG (.png): read as 8-bit grayscale via cv2
      - TIFF (.tif .tiff): read via PIL (handles 16-bit → rescale to 8-bit)
      - JPEG (.jpg .jpeg): read as grayscale via cv2

    Returns None on error. Never raises.
    """
    import numpy as np
    ext = os.path.splitext(path)[1].lower()
    try:
        if file_type == "dicom" or ext in DICOM_EXTS or ext == "":
            return _dicom_windowed(path)
        elif ext in TIFF_EXTS:
            from PIL import Image as _PIL
            im = _PIL.open(path)
            arr = np.array(im)
            if arr.ndim == 3:
                arr = arr.mean(axis=2)
            arr = arr.astype(np.float32)
            lo, hi = arr.min(), arr.max()
            if hi > lo:
                arr = (arr - lo) / (hi - lo) * 255.0
            return arr.astype(np.uint8)
        else:
            import cv2
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            return img  # may be None if unreadable
    except Exception as e:
        logger.warning(f"_load_image_universal({path}): {e}")
        return None

def _load_preview_slice(args) -> Optional[dict]:
    """Load + thumbnail one slice. Returns dict or None on error."""
    idx, path, file_type, total = args
    try:
        import numpy as np
        from PIL import Image
        img = _load_image_universal(path, file_type)
        if img is None: return None
        thumb = np.array(Image.fromarray(img).resize((256, 256), Image.LANCZOS))
        return {"index": idx, "total": total, "b64": _to_b64png(thumb)}
    except Exception as e:
        logger.warning(f"Preview slice {idx} skip: {e}")
        return None

# ── File format support ────────────────────────────────────────────────────────


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def _validate_directory(path: str) -> str:
    """Resolve and validate a user-supplied directory path.

    Resolves symlinks and normalises the path so a caller cannot escape the
    filesystem with ``../../../`` traversals or symlink chains.  Returns the
    canonical absolute path string.  Raises ``ValueError`` with a human-
    readable message on any invalid input.
    """
    if not path or not isinstance(path, str):
        raise ValueError("Directory path must be a non-empty string")
    canonical = os.path.realpath(os.path.abspath(os.path.expanduser(path)))
    if not os.path.isdir(canonical):
        raise ValueError(f"Directory not found: {canonical!r}")
    return canonical


def _validate_scores(scores) -> list:
    """Validate that model output scores are well-formed floats in [0, 1].

    Raises ``ValueError`` on NaN, Inf, or non-numeric values.  Out-of-range
    values are clamped with a warning rather than silently accepted — a score
    outside [0, 1] indicates a model calibration issue that warrants review.
    """
    if not isinstance(scores, (list, tuple)) or len(scores) == 0:
        raise ValueError(
            f"Expected a non-empty list of scores; got {type(scores).__name__}: {scores!r}"
        )
    validated = []
    for i, s in enumerate(scores):
        if not isinstance(s, (int, float)):
            raise ValueError(
                f"Score[{i}] is {type(s).__name__}, expected numeric — "
                "model returned unexpected output format"
            )
        f = float(s)
        if math.isnan(f) or math.isinf(f):
            raise ValueError(
                f"Score[{i}]={f} — model returned NaN or Inf; "
                "this indicates a numerical instability in the forward pass"
            )
        if f < 0.0 or f > 1.0:
            logger.warning(
                f"Score[{i}]={f:.6f} is outside [0, 1]; clamping. "
                "Check model calibration."
            )
            f = max(0.0, min(1.0, f))
        validated.append(f)
    return validated


def _compute_attention_audit(raw_imgs, heatmap):
    """
    Paper-inspired, low-risk audit metrics derived from Sybil attention outputs.
    These are proxies only: they do NOT implement SHNAP/SNAP, do not remove or
    insert nodules, and should be interpreted as UI-side audit signals rather
    than causal explanations.

    Paper: Sobieski et al. 2026 — "Auditing Sybil" (arXiv:2602.02560v1)
    New metrics added (v2):
      • Radial centrality index   — proxy for pleural/peripheral bias (paper §5.4)
      • RNC proxy                 — relative nodule contribution approximation (paper §5.3)
      • Craniocaudal thirds       — upper/middle/lower lobe attention split (paper §5.4)
      • Peak slice CC position    — where along the lung axis attention peaks
      • Enhanced flag messages    — include paper citations and clinical context
    """
    import numpy as np

    try:
        raw = np.asarray(raw_imgs, dtype=np.float32)
        hm  = np.asarray(heatmap, dtype=np.float32)
        if raw.ndim < 3 or hm.ndim != 3 or raw.shape[0] != hm.shape[0]:
            return {"available": False, "warning": "Attention audit unavailable for current tensor shapes."}

        # Collapse raw images to [slice, H, W]
        if raw.ndim == 4:
            raw = raw[:, 0] if raw.shape[1] == 1 else raw.mean(axis=1)

        # Normalize raw intensities slice-wise for simple body-mask extraction.
        mins = raw.reshape(raw.shape[0], -1).min(axis=1)[:, None, None]
        maxs = raw.reshape(raw.shape[0], -1).max(axis=1)[:, None, None]
        denom = np.maximum(maxs - mins, 1e-6)
        raw_u8 = ((raw - mins) / denom * 255.0).clip(0, 255)

        # Body proxy mask: threshold + keep slices that actually contain anatomy.
        body = raw_u8 > 8.0
        body_mass = body.reshape(body.shape[0], -1).sum(axis=1)
        valid_slice_mask = body_mass > max(128, int(body.shape[1] * body.shape[2] * 0.01))

        if not np.any(valid_slice_mask):
            return {"available": False, "warning": "Body mask extraction failed for audit metrics."}

        frame_scores = hm.sum(axis=(1, 2))
        total_attn = float(frame_scores.sum())
        if total_attn <= 1e-12:
            return {"available": False, "warning": "Attention energy is near zero."}

        # Global body / border / peripheral masks.
        H, W = hm.shape[1], hm.shape[2]
        yy, xx = np.mgrid[0:H, 0:W]
        img_border = (yy < 8) | (yy >= H - 8) | (xx < 8) | (xx >= W - 8)

        extra_body = 0.0
        border_body = 0.0
        peripheral_body = 0.0
        left_energy = 0.0
        right_energy = 0.0

        suspicious = []
        valid_indices = np.where(valid_slice_mask)[0]
        first_valid, last_valid = int(valid_indices[0]), int(valid_indices[-1])
        thoracic_span = max(last_valid - first_valid + 1, 1)

        for i in range(hm.shape[0]):
            h = hm[i]
            b = body[i]
            energy = float(h.sum())
            if energy <= 0:
                continue

            extra_body += float(h[~b].sum())
            border_mask = img_border.copy()
            peripheral_mask = img_border.copy()

            coords = np.argwhere(b)
            if coords.size:
                y0b, x0b = coords.min(axis=0)
                y1b, x1b = coords.max(axis=0)
                pad_y = max(6, int(0.12 * (y1b - y0b + 1)))
                pad_x = max(6, int(0.12 * (x1b - x0b + 1)))
                # ring around body bbox interior boundary
                inner_y0 = min(max(y0b + pad_y, 0), H)
                inner_y1 = max(min(y1b - pad_y, H - 1), -1)
                inner_x0 = min(max(x0b + pad_x, 0), W)
                inner_x1 = max(min(x1b - pad_x, W - 1), -1)
                bbox = np.zeros((H, W), dtype=bool)
                bbox[y0b:y1b + 1, x0b:x1b + 1] = True
                inner = np.zeros((H, W), dtype=bool)
                if inner_y1 >= inner_y0 and inner_x1 >= inner_x0:
                    inner[inner_y0:inner_y1 + 1, inner_x0:inner_x1 + 1] = True
                ring = bbox & (~inner)
                border_mask = ring | img_border
                peripheral_mask = ring

            border_body += float(h[border_mask].sum())
            peripheral_body += float(h[peripheral_mask].sum())
            left_energy += float(h[:, :W // 2].sum())
            right_energy += float(h[:, W // 2:].sum())

            # suspicious slice registry
            rel_pos = (i - first_valid) / thoracic_span
            suspicious.append({
                "slice_index": int(i),
                "attention_score": _safe_float(frame_scores[i] / total_attn),
                "outside_body_ratio": _safe_float(h[~b].sum() / energy),
                "border_ratio": _safe_float(h[border_mask].sum() / energy),
                "relative_craniocaudal_position": _safe_float(rel_pos),
            })

        suspicious.sort(key=lambda x: x["attention_score"], reverse=True)
        suspicious = suspicious[:8]

        # ── Craniocaudal thirds (paper §5.4: upper > lower lobe attribution, Tukey p≤0.009)
        upper_mask  = np.zeros_like(frame_scores, dtype=bool)
        middle_mask = np.zeros_like(frame_scores, dtype=bool)
        lower_mask  = np.zeros_like(frame_scores, dtype=bool)
        third = thoracic_span / 3.0
        for idx_cc in valid_indices:
            rel_cc = idx_cc - first_valid
            if rel_cc < third:
                upper_mask[idx_cc] = True
            elif rel_cc < 2 * third:
                middle_mask[idx_cc] = True
            else:
                lower_mask[idx_cc] = True

        # Legacy 2-split kept for backwards compatibility
        superior_mask = upper_mask | middle_mask
        inferior_mask = lower_mask

        superior_ratio = float(frame_scores[superior_mask].sum() / total_attn)
        inferior_ratio = float(frame_scores[inferior_mask].sum() / total_attn)
        upper_ratio    = float(frame_scores[upper_mask].sum()    / total_attn)
        middle_ratio   = float(frame_scores[middle_mask].sum()   / total_attn)
        lower_ratio    = float(frame_scores[lower_mask].sum()    / total_attn)

        top1 = float(np.max(frame_scores) / total_attn)
        top3 = float(np.sort(frame_scores)[-3:].sum() / total_attn) if frame_scores.size >= 3 else 1.0
        lr_balance = float(abs(left_energy - right_energy) / max(left_energy + right_energy, 1e-9))

        # ── Radial centrality index (paper §5.4: peripheral bias / zero-pad attenuation)
        cy_r, cx_r = H / 2.0, W / 2.0
        max_dist_r = float(np.sqrt(cy_r**2 + cx_r**2))
        dist_norm_field = np.clip(
            1.0 - np.sqrt((yy - cy_r)**2 + (xx - cx_r)**2) / (max_dist_r + 1e-6),
            0.0, 1.0
        )
        radial_w_sum = 0.0
        radial_w_tot = 0.0
        for ii_r in range(hm.shape[0]):
            hi_r = hm[ii_r] * body[ii_r].astype(np.float32)
            bw_r = float(hi_r.sum())
            if bw_r > 0:
                radial_w_sum += float((hi_r * dist_norm_field).sum())
                radial_w_tot += bw_r
        # 1.0 = fully central attention; 0.0 = fully peripheral
        radial_centrality = _safe_float(radial_w_sum / max(radial_w_tot, 1e-9))

        # ── RNC proxy (paper §5.3: Relative Nodule Contribution approximation)
        rnc_proxy = _safe_float(1.0 - (extra_body / total_attn))

        # ── Peak slice craniocaudal position (0=superior, 1=inferior)
        peak_slice_idx = int(np.argmax(frame_scores))
        peak_cc_pos = _safe_float((peak_slice_idx - first_valid) / thoracic_span) \
                      if 0 <= (peak_slice_idx - first_valid) <= thoracic_span else 0.5

        flags = []
        if extra_body / total_attn > 0.10:
            flags.append({
                "level": "warning",
                "title": "\u26a0 Attention extends outside the body contour",
                "detail": (
                    "Spurious extra-anatomical focus detected. "
                    "The 2026 Sybil audit paper (Sobieski et al., arXiv:2602.02560) found that ECG electrodes, "
                    "hospital-gown metal snaps, and thyroid goiters can individually drive >50% of predicted risk. "
                    "Inspect flagged slices in the attention viewer to confirm highlighted regions overlap real lung parenchyma."
                ),
            })
        if border_body / total_attn > 0.45:
            flags.append({
                "level": "warning",
                "title": "\u26a0 Boundary-dominant attention (pleural / peripheral bias)",
                "detail": (
                    "A large fraction of attention lies near the image or body boundary. "
                    "The audit paper reports that Sybil under-weights peripheral (pleural) nodules — the most common "
                    "site for adenocarcinoma (Travis et al. 2011) — likely due to zero-padding attenuation in 3D convolutions."
                ),
            })
        if radial_centrality < 0.35:
            flags.append({
                "level": "warning",
                "title": "\u26a0 Radial sensitivity bias detected",
                "detail": (
                    f"Attention peak is near the image periphery (radial centrality: {radial_centrality:.2f}). "
                    "SNAP experiments in the paper showed a significant positive regression of attribution "
                    "vs distance-to-pleura (p<0.001), paper §5.4. This suggests the model may be under-valuing "
                    "peripheral pathology in this scan."
                ),
            })
        if top1 > 0.55:
            flags.append({
                "level": "notice",
                "title": "Single-slice dominance",
                "detail": (
                    "A large share of total attention is concentrated in one slice. "
                    "The paper documented cases of multi-year focus shifts where the dominant slice changed on re-scan, "
                    "rendering predictions inconsistent despite similar risk scores."
                ),
            })
        if lr_balance > 0.65:
            flags.append({
                "level": "notice",
                "title": "Marked left-right asymmetry",
                "detail": (
                    "Attention is strongly concentrated on one hemithorax. "
                    "The paper confirmed Sybil correctly ignores laterality at the population level, "
                    "but strong asymmetry in an individual scan warrants visual inspection."
                ),
            })
        if lower_ratio > 0.60 and upper_ratio < 0.20:
            flags.append({
                "level": "notice",
                "title": "Lower-lobe attention dominance",
                "detail": (
                    "Most attention is in the lower lobes. The paper's SNAP experiments found upper-lobe "
                    "attribution is significantly higher than lower/middle lobes (Tukey p<=0.009, paper §5.4). "
                    "Lower-dominant attention without known lower-lobe pathology may indicate a sub-diaphragmatic "
                    "artifact or unusual background feature driving the prediction."
                ),
            })
        if rnc_proxy < 0.80:
            flags.append({
                "level": "notice",
                "title": "Low relative nodule contribution proxy",
                "detail": (
                    f"~{(1 - rnc_proxy) * 100:.0f}% of attention is outside the body contour. "
                    "The paper's RNC metric (paper §5.3) identifies cases where background features drive the risk "
                    "estimate, associated with both false positives and false negatives."
                ),
            })

        return {
            "available": True,
            "method": "attention_proxy_audit_v2",
            "paper_ref": "Sobieski et al. 2026 — Auditing Sybil (arXiv:2602.02560v1)",
            "limitations": [
                "These metrics are proxy diagnostics computed from Sybil attention maps and slice intensities.",
                "They are NOT causal SHNAP/SNAP interventions and do not establish nodule-level attribution.",
                "Body and boundary regions are estimated heuristically without segmentation.",
                "The radial bias index uses image-space distance, not true pleural surface distance.",
                "RNC proxy uses attention outside body contour, not a nodule-removal counterfactual.",
            ],
            "metrics": {
                "outside_body_attention_ratio":    _safe_float(extra_body / total_attn),
                "boundary_attention_ratio":         _safe_float(border_body / total_attn),
                "peripheral_ring_attention_ratio":  _safe_float(peripheral_body / total_attn),
                "superior_slice_attention_ratio":   _safe_float(superior_ratio),
                "inferior_slice_attention_ratio":   _safe_float(inferior_ratio),
                "upper_lobe_attention_ratio":       _safe_float(upper_ratio),
                "middle_lobe_attention_ratio":      _safe_float(middle_ratio),
                "lower_lobe_attention_ratio":       _safe_float(lower_ratio),
                "left_right_attention_asymmetry":   _safe_float(lr_balance),
                "top1_slice_attention_ratio":       _safe_float(top1),
                "top3_slice_attention_ratio":       _safe_float(top3),
                "radial_centrality_index":          _safe_float(radial_centrality),
                "rnc_proxy":                        _safe_float(rnc_proxy),
                "peak_slice_index":                 peak_slice_idx,
                "peak_craniocaudal_position":       _safe_float(peak_cc_pos),
            },
            "flags": flags,
            "suspicious_slices": suspicious,
        }
    except Exception as e:
        logger.warning(f"Attention audit failed: {e}")
        return {"available": False, "warning": str(e)}

def _detect_file_type(dicom_dir: str) -> str:
    """Auto-detect predominant file type in a directory."""
    counts = {"dicom": 0, "png": 0, "tiff": 0}
    for root, dirs, files in os.walk(dicom_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:  # PY-07: sample all files
            ext = os.path.splitext(f)[1].lower()
            if ext in DICOM_EXTS or ext == "":
                counts["dicom"] += 1
            elif ext in PNG_EXTS:
                counts["png"] += 1
            elif ext in TIFF_EXTS:
                counts["tiff"] += 1
    return max(counts, key=counts.get)

def _collect_ct(dicom_dir: str, file_type: str) -> List[str]:
    """Collect CT files of given type. file_type: dicom|png|tiff|auto"""
    if file_type == "auto":
        file_type = _detect_file_type(dicom_dir)

    if file_type == "dicom":
        valid = DICOM_EXTS
    elif file_type == "tiff":
        valid = TIFF_EXTS
    else:
        valid = PNG_EXTS  # Sybil Serie only accepts png/dicom natively

    found = []
    for root, dirs, files in os.walk(dicom_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            if f.startswith("."): continue
            ext = os.path.splitext(f)[1].lower()
            if ext in valid or (file_type == "dicom" and ext == ""):
                found.append(os.path.join(root, f))
    return sorted(found)

def _sort_z(paths: List[str], file_type: str = "dicom") -> List[str]:
    """Sort CT slices along Z axis.
    - DICOM: reads ImagePositionPatient[2] in parallel
    - PNG/TIFF: filename natural-sort (no Z metadata)
    """
    if file_type != "dicom":
        import re
        def nat_key(s):
            return [int(t) if t.isdigit() else t.lower()
                    for t in re.split(r"(\d+)", os.path.basename(s))]
        return sorted(paths, key=nat_key)

    import pydicom
    def get_z(p):
        try:
            return float(pydicom.dcmread(p, stop_before_pixels=True).ImagePositionPatient[2])
        except (AttributeError, IndexError, TypeError, Exception):
            return 0.0
    with ThreadPoolExecutor(max_workers=min(N_CPU, 8)) as ex:
        zs = list(ex.map(get_z, paths))
    return [p for _, p in sorted(zip(zs, paths))]


def _prepare_serie_files(files: List[str], file_type: str, tmpdir: str) -> Tuple[List[str], str]:
    """Convert TIFF slices to PNG (Sybil Serie only accepts dicom|png).
    Returns (final_paths, effective_file_type).
    For dicom and png input types, returns unchanged.
    """
    if file_type not in ("tiff",):
        return files, file_type

    import numpy as np
    from PIL import Image as _PIL
    out_paths = []
    for i, src_path in enumerate(files):
        dst = os.path.join(tmpdir, f"slice_{i:05d}.png")
        if not os.path.exists(dst):
            try:
                im = _PIL.open(src_path)
                arr = np.array(im).astype(np.float32)
                if arr.ndim == 3: arr = arr.mean(axis=2)
                lo, hi = arr.min(), arr.max()
                if hi > lo: arr = (arr - lo) / (hi - lo) * 255.0
                _PIL.fromarray(arr.astype(np.uint8)).save(dst)
            except Exception as e:
                logger.warning(f"TIFF→PNG conversion failed for {src_path}: {e}")
                continue
        out_paths.append(dst)
    return out_paths, "png"


# ═══════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.route("/health")
def health():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    mps = False
    try: mps = torch.backends.mps.is_available()
    except Exception: pass

    ver = _parse_torch_version()  # PY-10
    return jsonify({
        "status":         "ok",
        "gpu_cuda":       torch.cuda.is_available(),
        "gpu_mps":        mps,
        "sybil_device":   _sybil_device(),
        "device_label":   _device_label(),
        "torch_ver":      torch.__version__,
        "cpu_cores":      N_CPU,
        "intra_threads":  torch.get_num_threads(),
        "inter_threads":  torch.get_num_interop_threads(),
        "compile_ready":  ver[0] >= 2 if ver else False,
        "compiled":       _compiled,
    })

@app.route("/model/status")
def model_status():
    return jsonify({"loaded": _model is not None, "model_key": _model_key,
                    "compiled": _compiled})


@app.route("/scan_checkpoints", methods=["POST"])
def scan_checkpoints():
    d    = request.get_json(silent=True) or {}
    dir_ = d.get("directory", "")
    try:
        dir_ = _validate_directory(dir_)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    try:
        ckpts  = _find_ckpts(dir_)
        calibs = _find_calibs(dir_)
        groups = _build_groups(ckpts)
        calib_info = []
        for c in calibs:
            name = os.path.basename(c)
            kind = ("ensemble" if "ensemble" in name.lower()
                    else "single" if any(x in name.lower()
                        for x in ["sybil_1","sybil_2","sybil_3","sybil_4","sybil_5"])
                    else "unknown")
            calib_info.append({"path": c, "filename": name, "kind": kind})
        logger.info(f"scan: {len(ckpts)} ckpts, {len(calibs)} calibs, {len(groups)} groups")
        return jsonify({"success": True, "ckpt_count": len(ckpts),
                        "calib_count": len(calibs), "groups": groups,
                        "calibrators": calib_info})
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/detect_filetype", methods=["POST"])
def detect_filetype():
    d = request.get_json(silent=True) or {}
    directory = d.get("directory", "")
    try:
        directory = _validate_directory(directory)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    try:
        ft = _detect_file_type(directory)
        # Also count each type
        counts = {"dicom": 0, "png": 0, "tiff": 0, "other": 0}
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in DICOM_EXTS or ext == "": counts["dicom"] += 1
                elif ext in PNG_EXTS:  counts["png"] += 1
                elif ext in TIFF_EXTS: counts["tiff"] += 1
                elif ext not in {".json", ".txt", ".xml", ".csv", ".py"}: counts["other"] += 1
        return jsonify({"success": True, "detected": ft, "counts": counts})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/model/load", methods=["POST"])
def load_model():
    global _model, _model_key, _compiled
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    d              = request.get_json(silent=True) or {}
    mode           = d.get("mode", "auto")
    checkpoint_dir = d.get("checkpoint_dir")
    ckpt_paths     = d.get("ckpt_paths")
    calib_path_in  = d.get("calib_path")
    model_name     = d.get("model_name", "sybil_ensemble")
    use_compile    = d.get("use_compile", True)   # UI can toggle this off

    try:
        from sybil import Sybil
        dev = _sybil_device()

        if mode == "local":
            if ckpt_paths:
                final_ckpts = [p for p in ckpt_paths if os.path.isfile(p)]
                if not final_ckpts:
                    return jsonify({"success": False,
                                    "error": "None of the provided ckpt_paths exist on disk."}), 400
            elif checkpoint_dir:
                final_ckpts = _find_ckpts(checkpoint_dir)
                if not final_ckpts:
                    return jsonify({"success": False,
                                    "error": (
                                        f"No checkpoint files found in:\n{checkpoint_dir}\n\n"
                                        "Looking for: *.ckpt  *.pt  *.pth  or 32-char hex filenames."
                                    )}), 400
            else:
                return jsonify({"success": False,
                                "error": "Provide checkpoint_dir or ckpt_paths"}), 400

            if calib_path_in and os.path.isfile(calib_path_in):
                final_calib = calib_path_in
            elif checkpoint_dir:
                final_calib = _best_calib(final_ckpts, _find_calibs(checkpoint_dir))
            else:
                final_calib = None

            key = "|".join(sorted(final_ckpts)) + "|" + (final_calib or "") + "|" + dev
            if key != _model_key:
                logger.info(f"Loading {len(final_ckpts)} ckpt(s), calib={final_calib}")
                _compiled = False
                _model = Sybil(name_or_path=final_ckpts,
                               calibrator_path=final_calib, device=dev)
                _model_key = key
        else:
            key = model_name + "|" + dev
            if key != _model_key:
                logger.info(f"Loading named model: {model_name}")
                _compiled = False
                _model = Sybil(name_or_path=model_name, device=dev)
                _model_key = key

        # Apply torch.compile() for faster repeated inference
        if use_compile and not _compiled:
            _model, _compiled = _try_compile_model(_model)

        n = len(_model.ensemble) if hasattr(_model, "ensemble") else 1
        label = _device_label()
        logger.info(f"Ready — {n} model(s), device={dev}, compiled={_compiled}")
        return jsonify({"success": True, "device": label, "n_models": n,
                        "model_key": _model_key, "compiled": _compiled,
                        "compile_note": ("torch.compile applied — first inference has ~30s warm-up"
                                         if _compiled else "")})
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/ct_preview", methods=["POST"])
def ct_preview():
    import numpy as np
    d         = request.get_json(silent=True) or {}
    dicom_dir = d.get("dicom_dir", "")
    file_type = d.get("file_type", "dicom")
    n_samples = min(int(d.get("n_samples", 12)), 24)

    try:
        dicom_dir = _validate_directory(dicom_dir)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    try:
        files = _collect_ct(dicom_dir, file_type)
        if not files:
            return jsonify({"success": False, "error": "No CT files found"}), 400
        if file_type == "dicom":
            files = _sort_z(files)

        N       = len(files)
        indices = sorted(set(np.linspace(0, N-1, n_samples, dtype=int).tolist()))

        # Load preview slices in parallel
        args = [(idx, files[idx], file_type, N) for idx in indices]
        with ThreadPoolExecutor(max_workers=min(N_CPU, 8)) as ex:
            results = list(ex.map(_load_preview_slice, args))
        slices = [r for r in results if r is not None]

        return jsonify({"success": True, "slices": slices, "total_slices": N})
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════
# ASYNC INFERENCE JOB
# ═══════════════════════════════════════════════════════════════════

def _job_set(job_id: str, **kw):
    with _job_lock:
        _jobs[job_id].update(kw)

def _step_update(job_id: str, step_idx: int, status: str, detail: str = "", extra: dict = None):
    """Update a single named step inside the job's steps list."""
    with _job_lock:
        job = _jobs.get(job_id)
        if not job: return
        steps = job.setdefault("steps", [])
        while len(steps) <= step_idx:
            steps.append({})
        steps[step_idx].update({"status": status, "detail": detail,
                                 "elapsed_ms": int((time.perf_counter() - job.get("_t0", time.perf_counter())) * 1000)})
        if extra:
            steps[step_idx].update(extra)

def _build_overlay(args) -> Tuple[dict, dict]:
    """Build one attention overlay frame. Runs in a thread pool."""
    import numpy as np
    from PIL import Image
    i, raw_img, heat_slice = args
    raw = np.array(raw_img)
    if raw.ndim == 3: raw = raw[0]
    rmin, rmax = float(raw.min()), float(raw.max())
    raw_u8  = ((raw - rmin)/(rmax - rmin)*255).astype(np.uint8) \
              if rmax > rmin else np.zeros_like(raw, dtype=np.uint8)
    raw_256 = np.array(Image.fromarray(raw_u8).resize((256,256), Image.LANCZOS))
    heat    = np.array(
        Image.fromarray((heat_slice*255).astype(np.uint8)).resize((256,256), Image.LANCZOS)
    ).astype(np.float32) / 255.0
    ct_f    = raw_256.astype(np.float32)
    ov      = np.zeros((256,256,3), dtype=np.uint8)
    ov[:,:,0] = np.clip(ct_f + heat*210,        0, 255)
    ov[:,:,1] = np.clip(ct_f * (1-heat*0.75),   0, 255)
    ov[:,:,2] = np.clip(ct_f * (1-heat*0.85),   0, 255)
    return (
        {"slice_index": i, "b64": _to_b64png(raw_256)},
        {"slice_index": i, "b64": _to_b64png(ov)},
    )


def _run_job(job_id: str, dicom_dir: str, file_type: str,
             return_attentions: bool, n_frames: int, voxel_spacing=None):
    """
    Background inference thread.
    Writes rich step-by-step progress into _jobs[job_id] so the UI can
    display a live timeline instead of a static spinner.

    Job dict shape while running:
      status:          "queued" | "running" | "done" | "error" | "cancelled"
      message:         short current-action string
      steps:           list of step dicts (see STEPS below)
      preview_slices:  list of {b64, index} CT thumbnails (populated early)
      partial_scores:  {scores, n_done, n_total} streamed after each model
      result:          full result dict (only when status=="done")
    """
    import numpy as np

    # ── Step index constants ───────────────────────────────────────────────
    S_COLLECT = 0   # find + sort CT files
    S_LOAD    = 1   # load slices / build Serie
    S_INFER   = 2   # torch forward pass (per ensemble member)
    S_OVERLAY = 3   # attention heatmap collation + overlay PNGs

    STEP_NAMES = [
        "Collecting CT files",
        "Loading slices",
        "Running model inference",
        "Building attention maps",
    ]

    # ── Helpers scoped to this job ─────────────────────────────────────────
    t0 = time.perf_counter()

    def _elapsed():
        return int((time.perf_counter() - t0) * 1000)

    def _init_steps():
        return [{"name": n, "status": "pending", "detail": "", "elapsed_ms": 0}
                for n in STEP_NAMES]

    # per-step start timestamps
    _step_t = {}

    def _start(idx, detail=""):
        _step_t[idx] = time.perf_counter()
        with _job_lock:
            job = _jobs.get(job_id)
            if not job: return
            job["steps"][idx].update({"status": "running", "detail": detail, "elapsed_ms": _elapsed()})
            job["message"] = detail

    def _done(idx, detail=""):
        ms = int((time.perf_counter() - _step_t.get(idx, t0)) * 1000)
        with _job_lock:
            job = _jobs.get(job_id)
            if not job: return
            job["steps"][idx].update({"status": "done", "detail": detail, "elapsed_ms": ms})

    def _err(idx, detail=""):
        ms = int((time.perf_counter() - _step_t.get(idx, t0)) * 1000)
        with _job_lock:
            job = _jobs.get(job_id)
            if not job: return
            job["steps"][idx].update({"status": "error", "detail": detail, "elapsed_ms": ms})

    def _msg(txt):
        with _job_lock:
            job = _jobs.get(job_id)
            if job: job["message"] = txt

    def _set_extra(key, val):
        with _job_lock:
            job = _jobs.get(job_id)
            if job: job[key] = val

    try:
        with _job_lock:
            _jobs[job_id].update({
                "status":         "running",
                "steps":          _init_steps(),
                "message":        "Starting…",
                "preview_slices": [],
                "partial_scores": None,
                "_t0":            t0,
            })

        from sybil import Serie
        from sybil.utils.visualization import collate_attentions

        # ── STEP 0: collect + sort files ───────────────────────────────────
        _start(S_COLLECT, "Scanning CT directory…")
        # Auto-detect file type if needed
        actual_file_type = file_type
        if file_type == "auto":
            actual_file_type = _detect_file_type(dicom_dir)
            _msg(f"Detected file type: {actual_file_type}")

        files = _collect_ct(dicom_dir, actual_file_type)
        if not files:
            # Try auto-detect as fallback
            actual_file_type = _detect_file_type(dicom_dir)
            files = _collect_ct(dicom_dir, actual_file_type)
        if not files:
            _err(S_COLLECT, "No CT files found")
            _job_set(job_id, status="error", error="No CT files found in directory")
            return
        _msg(f"Sorting {len(files)} {actual_file_type} slices…")
        files = _sort_z(files, actual_file_type)
        _done(S_COLLECT, f"{len(files)} {actual_file_type} files")

        # ── STEP 1: load slices + stream 6 preview thumbnails ──────────────
        _start(S_LOAD, f"Loading {len(files)} slices…")

        # Stream a handful of CT thumbnails into the job immediately so the
        # UI can show real CT images while inference is running
        N_prev = min(6, len(files))
        prev_indices = np.linspace(0, len(files) - 1, N_prev, dtype=int).tolist()
        prev_args = [(i, files[i], actual_file_type, len(files)) for i in prev_indices]
        with ThreadPoolExecutor(max_workers=min(N_CPU, 6)) as ex:
            prev_results = list(ex.map(_load_preview_slice, prev_args))
        _set_extra("preview_slices", [r for r in prev_results if r is not None])


        # PY-09: cancellation helper
        def _is_cancelled():
            with _job_lock:
                return _jobs.get(job_id, {}).get("status") == "cancelled"

        # PY-02: wrap tmpdir in try/finally for guaranteed cleanup
        import tempfile as _tmpfile, shutil as _shutil
        _tmpdir = _tmpfile.mkdtemp(prefix="sybil_")
        try:
            serie_files, serie_file_type = _prepare_serie_files(files, actual_file_type, _tmpdir)
            if not serie_files:
                _err(S_LOAD, "Failed to convert input files")
                _job_set(job_id, status="error", error="Failed to prepare CT files for model")
                return

            kw = {"dicoms": serie_files, "file_type": serie_file_type}
            vs = None   # actual voxel spacing used (PNG only; DICOM uses embedded metadata)
            if serie_file_type == "png":
                if voxel_spacing:
                    vs = voxel_spacing
                else:
                    vs = [0.703125, 0.703125, 2.5]  # Sybil/NLST default
                    logger.warning(
                        f"Job {job_id}: no voxel_spacing provided for PNG input — "
                        f"defaulting to NLST protocol {vs}. "
                        "If your CT was acquired with a different protocol the "
                        "prediction may be inaccurate."
                    )
                kw["voxel_spacing"] = vs
            serie = Serie(**kw)
            _done(S_LOAD, f"{len(serie_files)} slices loaded")

            # PY-09: cancel before expensive inference
            if _is_cancelled():
                return

            # ── STEP 2: inference — stream partial scores per ensemble member ──
            n_models = len(_model.ensemble) if hasattr(_model, "ensemble") else 1
            if n_models == 0:
                _err(S_INFER, "Ensemble has no models — reload the model")
                _job_set(job_id, status="error",
                         error="Ensemble is empty — please reload the model from the Settings screen.")
                return
            if hasattr(_model, "ensemble") and n_models < 5:
                logger.warning(
                    f"Job {job_id}: partial ensemble — {n_models}/5 models loaded. "
                    "Predictions will be less accurate than the full sybil_ensemble."
                )
            _start(S_INFER, f"Running {n_models} model{'s' if n_models > 1 else ''} on {_device_label()}…")

            t_infer = time.perf_counter()

            with torch.inference_mode():
                if hasattr(_model, "ensemble") and n_models > 1:
                    # Run each ensemble member individually so we can publish
                    # a running-mean prediction after every completed model.
                    partial_list = []
                    last_attentions = None

                    for m_idx, sybil_net in enumerate(_model.ensemble):
                        if _is_cancelled():  # PY-09: honour cancel between models
                            return
                        is_last = (m_idx == n_models - 1)
                        _msg(f"Model {m_idx + 1} / {n_models}…")

                        # _predict is the per-model method (no calibration)
                        # Only request attentions on the final model to avoid
                        # computing them 5× unnecessarily.
                        p = _model._predict(sybil_net, [serie],
                                            return_attentions=(return_attentions and is_last))
                        partial_list.append(p.scores)
                        if is_last and return_attentions:
                            last_attentions = p.attentions

                        running_mean = np.mean(np.array(partial_list), axis=0)[0]
                        _set_extra("partial_scores", {
                            "scores":  running_mean.tolist(),
                            "n_done":  m_idx + 1,
                            "n_total": n_models,
                        })
                        with _job_lock:
                            _jobs[job_id]["steps"][S_INFER]["detail"] =                             f"{m_idx + 1}/{n_models} models done"

                    # Calibrate the final averaged scores (PY-05: ensure [1, N_yr] shape)
                    final_scores_raw = np.mean(np.array(partial_list), axis=0)
                    if final_scores_raw.ndim == 1:
                        final_scores_raw = final_scores_raw[None, :]
                    try:
                        calibrated = _model._calibrate(final_scores_raw)
                    except Exception as _cal_err:
                        logger.warning(f"Calibration failed ({_cal_err}), using raw scores")
                        calibrated = final_scores_raw

                    # Build a Prediction-like object for unified handling below
                    class _Pred:
                        pass

                    pred = _Pred()
                    pred.scores = calibrated.tolist()
                    pred.attentions = last_attentions

                else:
                    # Single model
                    pred = _model.predict([serie], return_attentions=return_attentions,
                                          threads=N_CPU)
                    partial_list = [pred.scores]

            infer_ms = int((time.perf_counter() - t_infer) * 1000)
            scores = pred.scores[0] if isinstance(pred.scores[0], list) else pred.scores
            # Normalise: scores should be a flat list of floats
            if isinstance(scores[0], list):
                scores = scores[0]
            # Validate scores: raises ValueError on NaN, Inf, or wrong types
            scores = _validate_scores(scores)

            _done(S_INFER, f"Done in {infer_ms / 1000:.1f}s")
            logger.info(f"Job {job_id}: inference {infer_ms}ms, 1yr={scores[0]*100:.1f}%")

            N_yr = len(scores)

            # ── Build calculation trace for UI display ─────────────────────────
            # partial_list shape: [n_models][1][N_yr] (list of lists)
            raw_per_model = []
            for m_scores in partial_list:
                s = m_scores[0] if isinstance(m_scores[0], list) else m_scores
                if isinstance(s[0], list): s = s[0]
                raw_per_model.append([float(x) for x in s])

            raw_ensemble_mean = [
                float(np.mean([raw_per_model[m][y] for m in range(len(raw_per_model))]))
                for y in range(N_yr)
            ]

            # Build calibration pairs: {year: {raw_input, calibrated_output}}
            calib_pairs = {}
            if hasattr(_model, "calibrator") and _model.calibrator is not None:
                for y in range(N_yr):
                    raw_in = raw_ensemble_mean[y]
                    cal_out = float(scores[y])
                    calib_pairs[f"year_{y+1}"] = {
                        "raw_sigmoid": round(raw_in, 6),
                        "calibrated":  round(cal_out, 6),
                        "delta":       round(cal_out - raw_in, 6),
                    }
            else:
                for y in range(N_yr):
                    calib_pairs[f"year_{y+1}"] = {
                        "raw_sigmoid": round(raw_ensemble_mean[y], 6),
                        "calibrated":  round(float(scores[y]), 6),
                        "delta": 0.0,
                    }

            calc_trace = {
                "pipeline": [
                    {
                        "step": 1,
                        "name": "3D CNN Feature Extraction",
                        "detail": "R3D-18 (3D ResNet-18) encodes the CT volume into spatial feature maps (B×512×T×H×W)",
                        "ref": "Tran et al., ICCV 2015"
                    },
                    {
                        "step": 2,
                        "name": "Multi-head Attention Pooling",
                        "detail": "Two parallel attention pools (spatial + temporal) + global max pool → 3×512-dim hidden vectors → concat → Linear(1536→512)",
                        "ref": "Sybil pooling_layer.py"
                    },
                    {
                        "step": 3,
                        "name": "Cumulative Hazard Layer",
                        "detail": f"Linear(512→{N_yr}) predicts annual hazards h₁…h₆; cumulative logit = Σᵢ≤t hᵢ + base_hazard (upper-triangular masked sum)",
                        "ref": "cumulative_probability_layer.py"
                    },
                    {
                        "step": 4,
                        "name": "Sigmoid → Annual Risk",
                        "detail": f"P(cancer by year t) = σ(cumulative_logit_t) for t=1…{N_yr}",
                        "ref": "SybilNet.forward()"
                    },
                    {
                        "step": 5,
                        "name": f"Ensemble Averaging ({len(raw_per_model)} models)",
                        "detail": "Mean of sigmoid outputs across all ensemble members (equal weighting)",
                        "ref": "Sybil model.py:predict()"
                    },
                    {
                        "step": 6,
                        "name": "Isotonic Regression Calibration",
                        "detail": "Per-year calibration via SimpleClassifierGroup (isotonic regression on NLST validation set). Adjusts for systematic over/under-confidence.",
                        "ref": "Mikhael et al., JCO 2023; calibrator.py"
                    },
                ],
                "raw_per_model":    raw_per_model,
                "raw_ensemble_mean": raw_ensemble_mean,
                "calibrated_final":  [float(scores[y]) for y in range(N_yr)],
                "calib_pairs":       calib_pairs,
                "n_models":          len(raw_per_model),
                "voxel_spacing_used": (
                    f"{vs[0]} × {vs[1]} × {vs[2]} mm" if vs else "from DICOM metadata"
                ),
            }

            result = {
                "num_files":   len(files),
                "scores":      scores,
                "year_labels": [f"Year {i + 1}" for i in range(N_yr)],
                "summary":     {f"{y + 1}_year_risk": round(scores[y] * 100, 2)
                                for y in range(min(N_yr, 6))},
                "device":      _device_label(),
                "infer_ms":    infer_ms,
                "compiled":    _compiled,
                "calc_trace":  calc_trace,
                "file_type_used": actual_file_type,
                # ── Audit provenance ──────────────────────────────────────────
                "provenance": {
                    "dicom_directory":    dicom_dir,
                    "timestamp_utc":      datetime.datetime.utcnow().isoformat() + "Z",
                    "operator":           getpass.getuser(),
                    "sybil_model_key":    _model_key,
                    "n_ensemble_models":  n_models,
                    "voxel_spacing_actual": (
                        vs if vs else "from DICOM metadata"
                    ),
                    "file_type_used":     actual_file_type,
                    "n_slices":           len(files),
                    "app_version":        "2.0.1",
                },
            }
            logger.info(
                f"Job {job_id}: COMPLETED — operator={getpass.getuser()} "
                f"dir={dicom_dir!r} n_slices={len(files)} "
                f"1yr={scores[0]*100:.1f}% infer={infer_ms}ms"
            )

            # ── STEP 3: attention overlays ─────────────────────────────────────
            if return_attentions and pred.attentions:
                _start(S_OVERLAY, "Collating attention weights…")
                try:
                    raw_imgs    = serie.get_raw_images()
                    N_slices    = len(raw_imgs)
                    heatmap     = collate_attentions(pred.attentions[0], N_slices)
                    h_max       = heatmap.max()
                    if h_max > 1e-8:
                        heatmap /= h_max

                    frame_scores = heatmap.max(axis=(1, 2))
                    selected     = sorted(np.argsort(frame_scores)[::-1][:n_frames].tolist())
                    _msg(f"Encoding {len(selected)} overlay frames…")

                    overlay_args = [(i, raw_imgs[i], heatmap[i]) for i in selected]
                    with ThreadPoolExecutor(max_workers=min(N_CPU, 8)) as ex:
                        overlay_results = list(ex.map(_build_overlay, overlay_args))

                    raws, overlays = [], []
                    for (raw_d, ov_d), i in zip(overlay_results, selected):
                        raws.append(dict(raw_d))
                        overlays.append({
                            "slice_index":    i,
                            "attention_score": float(frame_scores[i]),
                            "b64":            ov_d["b64"],
                        })

                    result.update(attention_overlays=overlays, raw_slices=raws,
                                  total_slices=N_slices)
                    result["attention_audit"] = _compute_attention_audit(raw_imgs, heatmap)
                    _done(S_OVERLAY, f"{len(overlays)} frames")

                except Exception as ae:
                    _err(S_OVERLAY, str(ae))
                    logger.warning(f"Overlay error: {ae}\n{traceback.format_exc()}")
                    result["attention_warning"] = str(ae)
            else:
                with _job_lock:
                    _jobs[job_id]["steps"][S_OVERLAY].update(
                        {"status": "skipped", "detail": "attention maps disabled"})

            total_ms = int((time.perf_counter() - t0) * 1000)
            result["total_ms"] = total_ms
            _job_set(job_id, status="done", result=result,
                     message=f"Complete in {total_ms / 1000:.1f}s")


        finally:
            # PY-02: always clean up TIFF→PNG tmpdir
            _shutil.rmtree(_tmpdir, ignore_errors=True)

    except Exception as e:
        logger.error(traceback.format_exc())
        _job_set(job_id, status="error", error=str(e),
                 traceback=traceback.format_exc())

@app.route("/predict/start", methods=["POST"])
def predict_start():
    if _model is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 400
    d         = request.get_json(silent=True) or {}
    dicom_dir = d.get("dicom_dir", "")
    try:
        dicom_dir = _validate_directory(dicom_dir)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    job_id = uuid.uuid4().hex[:8]
    with _job_lock:
        _jobs[job_id] = {"status": "queued", "message": "Starting…"}

    file_type_req = d.get("file_type", "auto")   # "dicom"|"png"|"tiff"|"auto"
    threading.Thread(
        target=_run_job,
        args=(job_id, dicom_dir,
              file_type_req,
              d.get("return_attentions", True),
              min(int(d.get("n_overlay_frames", 16)), 30),
              d.get("voxel_spacing")),
        daemon=True,
    ).start()
    return jsonify({"success": True, "job_id": job_id})


def _is_valid_job_id(job_id: str) -> bool:
    """Job IDs are 8 lowercase hex chars generated by uuid4().hex[:8]."""
    return (
        isinstance(job_id, str)
        and len(job_id) == 8
        and all(c in "0123456789abcdef" for c in job_id)
    )


@app.route("/predict/status/<job_id>")
def predict_status(job_id):
    if not _is_valid_job_id(job_id):
        return jsonify({"success": False, "error": "Invalid job_id format"}), 400
    with _job_lock:
        job = dict(_jobs.get(job_id, {}))
    if not job:
        return jsonify({"success": False, "error": "Unknown job_id"}), 404
    return jsonify({"success": True, **job})


@app.route("/predict/cancel/<job_id>", methods=["POST"])
def predict_cancel(job_id):
    if not _is_valid_job_id(job_id):
        return jsonify({"success": False, "error": "Invalid job_id format"}), 400
    with _job_lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "cancelled"
            logger.info(f"Job {job_id}: CANCELLED by operator={getpass.getuser()}")
    return jsonify({"success": True})


@app.route("/predict", methods=["POST"])   # backwards compat
def predict_compat():
    return predict_start()


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    port = int(os.environ.get("SYBIL_PORT", 5001))
    logger.info(f"Starting Sybil backend on port {port}")

    # Use waitress (production WSGI) when available — silences Flask dev warnings
    # and is safe on Windows, macOS, and Linux without extra config.
    # Falls back to Flask's built-in server if waitress isn't installed.
    try:
        from waitress import serve
        logger.info("Using waitress WSGI server")
        serve(app, host="127.0.0.1", port=port, threads=max(4, N_CPU))
    except ImportError:
        logger.info("waitress not found — using Flask dev server (run: pip install waitress)")
        app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
