from __future__ import annotations

import hashlib
import json
import threading
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .parsing import digest
from .text import prepare_text as prepare_text


def mean_pool(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if hidden.ndim != 3 or hidden.shape[:2] != mask.shape:
        raise ValueError("Unexpected encoder/mask shape")
    weighted = hidden * mask[..., None]
    pooled = weighted.sum(axis=1) / np.maximum(mask.sum(axis=1, keepdims=True), 1)
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    if not np.isfinite(pooled).all() or (norms < 1e-8).any():
        raise ValueError("Nonfinite/zero embedding")
    return (pooled / norms).astype(np.float32)


class E5Encoder:
    dimension = 384

    def __init__(self, model_dir: Path, threads: int = 2):
        self.model_dir = model_dir.resolve(strict=True)
        receipt = json.loads(
            (self.model_dir / "download_manifest.json").read_text(encoding="utf-8")
        )
        if receipt["repo"] != "intfloat/multilingual-e5-small":
            raise ValueError("Only the reviewed E5-small snapshot is supported")
        self.model_file = receipt.get("model_file", "onnx/model.onnx")
        if self.model_file not in {"onnx/model.onnx", "onnx/model_qint8_avx512_vnni.onnx"}:
            raise ValueError("Unreviewed model graph")
        for name in (self.model_file, "tokenizer.json"):
            path = self.model_dir / name
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != receipt["files"][name]["sha256"]:
                raise ValueError("Local embedding model checksum mismatch")
        self.descriptor = {
            "model": receipt["repo"],
            "revision": receipt["revision"],
            "onnx_sha256": receipt["files"][self.model_file]["sha256"],
            "tokenizer_sha256": receipt["files"]["tokenizer.json"]["sha256"],
            "dimension": self.dimension,
            "max_tokens": 512,
            "pooling": "masked_mean_l2",
            "prefixes": ["query: ", "passage: "],
            "passage_title": True,
            "normalization": "nfc-strip-soft-hyphen-collapse-whitespace-v1",
            "onnxruntime": version("onnxruntime"),
            "tokenizers": version("tokenizers"),
            "provider": "CPUExecutionProvider",
            "precision": "int8" if "qint8" in self.model_file else "fp32",
        }
        self.fingerprint = digest(json.dumps(self.descriptor, sort_keys=True))
        self.threads = threads
        self._session = self._tokenizer = None
        self._lock = threading.RLock()
        self.texts_encoded = 0
        self.truncated_texts = 0

    def _load(self):
        if self._session is not None:
            return
        import onnxruntime as ort
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        tokenizer.enable_truncation(max_length=512)
        tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        session = ort.InferenceSession(
            str(self.model_dir / self.model_file),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer, self._session = tokenizer, session

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        with self._lock:
            self._load()
            prefix = "query: " if query else "passage: "
            encodings = self._tokenizer.encode_batch([prefix + prepare_text(t) for t in texts])
            self.texts_encoded += len(encodings)
            self.truncated_texts += sum(bool(e.overflowing) for e in encodings)
            arrays = {
                "input_ids": np.asarray([e.ids for e in encodings], dtype=np.int64),
                "attention_mask": np.asarray([e.attention_mask for e in encodings], dtype=np.int64),
                "token_type_ids": np.asarray([e.type_ids for e in encodings], dtype=np.int64),
            }
            feed = {item.name: arrays[item.name] for item in self._session.get_inputs()}
            hidden = self._session.run(None, feed)[0]
            vectors = mean_pool(hidden, arrays["attention_mask"])
            if vectors.shape != (len(texts), self.dimension):
                raise ValueError("Embedding dimension differs from model descriptor")
            return vectors
