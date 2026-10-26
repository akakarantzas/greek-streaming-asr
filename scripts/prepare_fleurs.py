"""Download a FLEURS Greek (el_gr) split and write a JSONL manifest for it.

FLEURS is published by Google as google/fleurs on Hugging Face under CC BY 4.0.
The audio is 16 kHz mono WAV (32-bit float), so no conversion is needed.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import IO, Sequence


BASE_URL = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/el_gr"
SAMPLE_RATE = 16000
# Seconds of silence on the connection before giving up; a slow but moving download is fine.
TIMEOUT_S = 60
SPLITS = ("train", "dev", "test")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a FLEURS Greek (el_gr) split and write a JSONL manifest.",
    )
    parser.add_argument("--split", default="test", choices=SPLITS, help="FLEURS split to download.")
    parser.add_argument(
        "--output-dir",
        default="data/fleurs",
        help=(
            "Audio goes to <output-dir>/<split>/, the manifest to <output-dir>/el_gr_<split>.jsonl "
            "(el_gr_<split>_first<N>.jsonl with --limit)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Keep only the first N utterances, for a quick smoke run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        print("--limit must be positive", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    audio_dir = output_dir / args.split
    # A limited run gets its own name, so it never overwrites the full split's manifest.
    suffix = f"_first{args.limit}" if args.limit else ""
    manifest_path = output_dir / f"el_gr_{args.split}{suffix}.jsonl"
    try:
        with urllib.request.urlopen(f"{BASE_URL}/{args.split}.tsv", timeout=TIMEOUT_S) as response:
            rows = parse_tsv(response.read().decode("utf-8"))
        rows = rows[: args.limit] if args.limit else rows

        audio_dir.mkdir(parents=True, exist_ok=True)
        missing = {row["file_name"] for row in rows} - {path.name for path in audio_dir.glob("*.wav")}
        if missing:
            print(f"Downloading FLEURS el_gr {args.split} audio for {len(missing)} files...")
            with urllib.request.urlopen(f"{BASE_URL}/audio/{args.split}.tar.gz", timeout=TIMEOUT_S) as response:
                found = extract_wavs(response, missing, audio_dir)
            if found != missing:
                raise RuntimeError(f"{len(missing - found)} WAV files listed in the TSV are not in the archive")

        write_manifest(rows, manifest_path, audio_subdir=args.split)
    except (OSError, ValueError, RuntimeError, tarfile.TarError) as exc:
        print(f"FLEURS download failed: {exc}", file=sys.stderr)
        return 1

    total_s = sum(row["duration"] for row in rows)
    print(f"Wrote {len(rows)} utterances ({total_s / 60:.1f} min) to {manifest_path}")
    return 0


def parse_tsv(text: str) -> list[dict]:
    """FLEURS columns: id, file name, raw text, normalized text, characters, num samples, gender."""
    rows: list[dict] = []
    for line_number, line in enumerate(text.split("\n"), start=1):
        line = line.rstrip("\r")
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) != 7:
            raise ValueError(f"FLEURS TSV line {line_number}: expected 7 columns, got {len(fields)}")
        file_name, raw_text, num_samples = fields[1], fields[2], fields[5]
        # Remote data: only plain file names, so nothing can be written outside the audio folder.
        if Path(file_name).name != file_name or not file_name.endswith(".wav"):
            raise ValueError(f"FLEURS TSV line {line_number}: unexpected file name {file_name!r}")
        rows.append(
            {
                "file_name": file_name,
                "text": raw_text,
                "duration": int(num_samples) / SAMPLE_RATE,
            }
        )
    return rows


def extract_wavs(archive_file: IO[bytes], wanted: set[str], audio_dir: Path) -> set[str]:
    """Stream a .tar.gz and write only the wanted WAVs, named by base name, never by archive path."""
    found: set[str] = set()
    with tarfile.open(fileobj=archive_file, mode="r|gz") as archive:
        for member in archive:
            name = Path(member.name).name
            if not member.isfile() or name not in wanted:
                continue
            source = archive.extractfile(member)
            # Write then rename, so an interrupted download never leaves a truncated WAV behind.
            partial_path = audio_dir / f"{name}.part"
            partial_path.write_bytes(source.read())
            partial_path.replace(audio_dir / name)
            found.add(name)
            if found == wanted:
                break
    return found


def write_manifest(rows: list[dict], manifest_path: Path, audio_subdir: str) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for row in rows:
            entry = {
                # Relative to the manifest's folder, which load_jsonl_manifest resolves against.
                "audio_filepath": f"{audio_subdir}/{row['file_name']}",
                "text": row["text"],
                "duration": row["duration"],
            }
            manifest_file.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
