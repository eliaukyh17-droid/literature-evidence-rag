"""Download a pinned public E5 ONNX snapshot. No credentials or corpus are read."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

from literature_kb.config import ROOT

REPO = "intfloat/multilingual-e5-small"
FILES = ["tokenizer.json", "config.json", "tokenizer_config.json", "README.md"]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(path, metadata):
    if path.stat().st_size != metadata["size"]:
        raise ValueError(f"Size mismatch: {path.name}")
    if metadata.get("lfs"):
        if sha256(path) != metadata["lfs"]["sha256"]:
            raise ValueError(f"LFS SHA256 mismatch: {path.name}")
    else:
        # Standard Hub files are Git blob objects, not SHA256-addressed LFS objects.
        git_hash = hashlib.sha1(f"blob {path.stat().st_size}\0".encode(), usedforsecurity=False)
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                git_hash.update(block)
        if git_hash.hexdigest() != metadata["blobId"]:
            raise ValueError(f"Git blob mismatch: {path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--revision", default="main", help="Resolve once to an immutable Hub commit"
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / ".local/models")
    parser.add_argument("--variant", choices=["fp32", "int8"], default="fp32")
    args = parser.parse_args()
    url = f"https://huggingface.co/api/models/{REPO}/revision/{quote(args.revision, safe='')}?blobs=true"
    with urlopen(
        Request(url, headers={"User-Agent": "literature-kb-public-model-download"}), timeout=30
    ) as response:
        info = json.load(response)
    revision = info["sha"]
    files = {item["rfilename"]: item for item in info["siblings"]}
    model_file = (
        "onnx/model.onnx" if args.variant == "fp32" else "onnx/model_qint8_avx512_vnni.onnx"
    )
    target = args.output_root / (
        "multilingual-e5-small-" + revision[:12] + ("-int8" if args.variant == "int8" else "")
    )
    target.mkdir(parents=True, exist_ok=True)
    receipt = {
        "repo": REPO,
        "revision": revision,
        "files": {},
        "credentials_read": False,
        "private_data_uploaded": False,
        "runtime": "onnx-" + args.variant + "-cpu",
    }
    if args.variant == "int8":
        receipt["model_file"] = model_file
    for name in [model_file, *FILES]:
        meta = files[name]
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            pending = path.with_name(path.name + ".pending")
            file_url = f"https://huggingface.co/{REPO}/resolve/{revision}/{name}?download=true"
            print(
                json.dumps({"event": "download_start", "file": name, "bytes": meta["size"]}),
                flush=True,
            )
            with (
                urlopen(
                    Request(
                        file_url, headers={"User-Agent": "literature-kb-public-model-download"}
                    ),
                    timeout=45,
                ) as response,
                pending.open("wb") as out,
            ):
                while block := response.read(1024 * 1024):
                    out.write(block)
            verify(pending, meta)
            pending.rename(path)
        verify(path, meta)
        receipt["files"][name] = {
            "size": path.stat().st_size,
            "sha256": sha256(path),
            "hub_blob_id": meta["blobId"],
        }
        print(
            json.dumps({"event": "download_verified", "file": name, "bytes": path.stat().st_size}),
            flush=True,
        )
    receipt_file = target / "download_manifest.json"
    if receipt_file.exists() and json.loads(receipt_file.read_text(encoding="utf-8")) != receipt:
        raise RuntimeError("Existing model receipt differs")
    if not receipt_file.exists():
        receipt_file.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "model_download_complete",
                "model_dir": str(target.resolve()),
                "revision": revision,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
