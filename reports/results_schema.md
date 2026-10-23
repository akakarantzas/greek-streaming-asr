# Results Schema

Fields written by the scripts today, with fields that are not implemented yet listed separately.

## Per-file CSV (`scripts/run_batch_eval.py --output`)

- `audio_filepath`
- `reference_text`
- `hypothesis_text`
- `language`
- `language_readiness`
- `wer` (accents counted; empty when the language uses CER)
- `wer_no_accents`
- `cer` (empty for Greek, which is scored with WER)
- `latency_ms`
- `gpu_memory_allocated_mb`
- `duration` (from the manifest, when present)
- `model_id_or_path`
- `device` (resolved: `cuda` or `cpu`)
- `warnings` (JSON list)

## Summary JSON (`scripts/run_batch_eval.py --summary-output`)

- `language`
- `num_files`
- `metric_name` (`wer` for Greek)
- `metric_value` (accents counted, headline)
- `metric_value_no_accents`
- `avg_latency_ms`
- `p50_latency_ms`
- `p95_latency_ms`
- `substitutions`
- `deletions`
- `insertions`
- `model_id_or_path`
- `target_lang`
- `language_readiness`
- `warnings`
- `manifest_path`
- `device` (resolved: `cuda` or `cpu`)
- `gpu_name` (null on CPU)

## Environment check JSON (`scripts/verify_environment.py --json-output`)

- `python_version`
- `package_versions` (object keyed by `torch`, `torchaudio`, `nemo`, `huggingface_hub`)
- `cuda_available`
- `gpu_name`
- `gpu_total_memory_gb`
- `gpu_memory_allocated_mb`
- `gpu_memory_reserved_mb`
- `nvidia_smi_available`
- `checks`
- `overall_status` (`PASS` or `FAIL`)

## Not implemented yet

- Summary: `model_revision`, `total_words`, `total_chars`, `normalization`, `gpu_memory_allocated_mb_max`
- Environment check: `platform`, `torch_cuda_version`, `driver_version`, `warnings`
