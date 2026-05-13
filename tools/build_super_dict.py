"""Build the frozen Super Dict (super_dict.zdict) for ZPackR v2.0.

One-time offline build.  Requires:
  - datasets (HuggingFace) for GLUE corpus
  - /usr/share/dict/words (Unix word list)
  - zstandard

Usage:
    python tools/build_super_dict.py [--output packr/super_dict.zdict]
"""

import os
import sys
import argparse


DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(__file__), "..", "packr", "super_dict.zdict"
)
DICT_SIZE = 131072  # 128 KB — hits the roadmap's ~100-200 KB target
MAX_SAMPLE_SIZE = 4096  # chunk large texts for zstd dictionary training
MAX_TOTAL_BYTES = 128 * 1024 * 1024  # cap total corpus at 128 MB


def _ensure_imports():
    project_root = os.path.join(os.path.dirname(__file__), "..")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def build_super_dict(output_path: str):
    _ensure_imports()

    import zstandard as zstd

    samples = []
    total_bytes = 0

    # ── 1. English word list ──
    words_path = "/usr/share/dict/words"
    if os.path.exists(words_path):
        print(f"Loading word list from {words_path} ...")
        with open(words_path, "r", encoding="utf-8", errors="ignore") as f:
            word_text = f.read()
        samples.append(word_text.encode("utf-8"))
        print(f"  {len(samples[-1])} bytes from word list")
    else:
        print(f"Warning: {words_path} not found, skipping word list")

    # ── 2. GLUE training corpus ──
    try:
        from datasets import load_dataset
    except ImportError:
        print("Warning: datasets not installed, skipping GLUE corpus")
        load_dataset = None

    if load_dataset is not None:
        glue_tasks = [
            ("sst2", "sentence"),
            ("mnli", "premise"),
            ("mnli", "hypothesis"),
            ("qnli", "question"),
            ("qnli", "sentence"),
            ("qqp", "question1"),
            ("qqp", "question2"),
            ("rte", "sentence1"),
            ("rte", "sentence2"),
            ("mrpc", "sentence1"),
            ("mrpc", "sentence2"),
            ("cola", "sentence"),
            ("stsb", "sentence1"),
            ("stsb", "sentence2"),
        ]

        seen_pairs = set()
        total_bytes = sum(len(s) for s in samples)

        for task, field in glue_tasks:
            if total_bytes >= MAX_TOTAL_BYTES:
                break
            pair_key = (task, field)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            try:
                print(f"Loading GLUE/{task}.{field} ...")
                ds = load_dataset("glue", task, split="train", trust_remote_code=True)
            except Exception as e:
                print(f"  Skipping {task}: {e}")
                continue

            texts = []
            for row in ds:
                text = str(row.get(field, ""))
                if text and text != "None":
                    texts.append(text)

            full_text = "\n".join(texts)
            text_bytes = full_text.encode("utf-8")

            # Chunk large texts for zstd dictionary training
            for start in range(0, len(text_bytes), MAX_SAMPLE_SIZE):
                chunk = text_bytes[start:start + MAX_SAMPLE_SIZE]
                if len(chunk) < 64:
                    continue
                samples.append(chunk)
                total_bytes += len(chunk)
                if total_bytes >= MAX_TOTAL_BYTES:
                    break

            print(f"  {len(text_bytes)} bytes from {task}.{field}")

    if not samples:
        print("Error: No training samples collected.")
        sys.exit(1)

    print(f"\nTotal corpus: {total_bytes:,} bytes in {len(samples)} samples")
    print(f"Training zstd dictionary ({DICT_SIZE:,} bytes) ...")

    # ── 3. Train the dictionary ──
    dict_obj = zstd.train_dictionary(DICT_SIZE, samples)
    dict_data = dict_obj.as_bytes()

    # ── 4. Save ──
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(dict_data)

    print(f"Saved Super Dict: {output_path} ({len(dict_data):,} bytes)")

    # ── 5. Validate ──
    cctx = zstd.ZstdCompressor(level=3, dict_data=zstd.ZstdCompressionDict(dict_data))

    for label, sample in [("word list", samples[0])] + [
        (f"GLUE chunk {i}", s) for i, s in enumerate(samples[1:6])
    ]:
        if len(sample) < 64:
            continue
        compressed = cctx.compress(sample)
        ratio = len(sample) / max(len(compressed), 1)
        print(f"  {label}: {len(sample)} -> {len(compressed)} bytes (ratio {ratio:.2f})")

    print("\nDone.  Commit super_dict.zdict to version control.")


def main():
    parser = argparse.ArgumentParser(description="Build ZPackR Super Dict")
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT,
        help=f"Output path for super_dict.zdict (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()
    build_super_dict(args.output)


if __name__ == "__main__":
    main()
