"""Source lexical codec for the Mamba-3 / Llama 3.1 IQ checkpoint.

Public copies are separately distributed under the Llama 3.1 license.
Cross-check their tokenization pipeline before using the public codec.
This is not a cryptographic attestation against the restricted donor file.
"""
from pathlib import Path
from hashlib import sha256
import argparse
import json

PUBLIC = "Xenova/Meta-Llama-3.1-Tokenizer"
REFERENCE = "NousResearch/Meta-Llama-3.1-8B"
OFFICIAL = "meta-llama/Llama-3.1-8B"
SPECIAL = {"<|begin_of_text|>": 128000, "<|end_of_text|>": 128001,
           "<|start_header_id|>": 128006, "<|end_header_id|>": 128007,
           "<|eot_id|>": 128009}


def load_source_tokenizer(directory=None, *, official=False):
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    if directory is not None:
        path = Path(directory) / "tokenizer.json"
        source = str(path)
    elif official:
        path = Path(hf_hub_download(OFFICIAL, "tokenizer.json"))
        source = OFFICIAL
    else:
        path = Path(hf_hub_download(PUBLIC, "tokenizer.json", token=False))
        reference_path = Path(hf_hub_download(REFERENCE, "tokenizer.json", token=False))
        candidate = json.loads(path.read_text(encoding="utf-8"))
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        for field in ("model", "normalizer", "pre_tokenizer", "decoder"):
            if candidate.get(field) != reference.get(field):
                raise RuntimeError(f"Public source tokenizer mismatch: {field}")
        a, b = Tokenizer.from_file(str(path)), Tokenizer.from_file(str(reference_path))
        if a.get_vocab() != b.get_vocab():
            raise RuntimeError("Public source tokenizer vocabulary mappings differ")
        source = PUBLIC + " (cross-checked against " + REFERENCE + ")"
    tok = Tokenizer.from_file(str(path))
    if tok.get_vocab_size(with_added_tokens=True) != 128256 or any(
        tok.token_to_id(k) != v for k, v in SPECIAL.items()
    ):
        raise RuntimeError("Source tokenizer contract mismatch")
    print("Tokenizer:", source, "SHA256:", sha256(path.read_bytes()).hexdigest(), flush=True)
    return tok


def encode_prompt(tokenizer, text):
    # Original base model uses a BOS token; avoid alternate postprocessors.
    return [SPECIAL["<|begin_of_text|>"]] + tokenizer.encode(
        text, add_special_tokens=False
    ).ids


def decode_text(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode-ids", nargs="+", type=int, required=True)
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--official-tokenizer", action="store_true")
    args = parser.parse_args()
    tok = load_source_tokenizer(args.tokenizer_dir, official=args.official_tokenizer)
    print("Decoded text:", repr(decode_text(tok, args.decode_ids)), flush=True)
