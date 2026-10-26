# Offline diarization PoC

This tool evaluates anonymous **who spoke when**, preserving overlap. It is
independent of the HTTP service, model registry and transcription pipeline.
Read the [approved scope](../docs/plans/2026-10-23-nemotron-diarization-poc.md)
and [measured decision](../docs/plans/2026-10-26-nemotron-diarization-results.md)
before extending it. No transcript-attributed accuracy is implied.

## Isolated setup

From the task worktree, use a new environment and a retained evidence directory.
Do not run `uv sync` against the running checkout's environment.

```bash
export UV_PROJECT_ENVIRONMENT=/private/tmp/nemotron-poc/venv
uv sync --frozen --dev
export EVIDENCE=/absolute/path/to/retained/nemotron-poc
mkdir -p "$EVIDENCE"
cp uv.lock "$EVIDENCE/uv.lock"
```

Use the pinned Nemotron checkpoint
`mlx-community/Nemotron-3-Diarization@59ed2dbfc1346dcea9d423c71306a3a2499c568f`.
The model card/config/weights can be fetched explicitly before offline evaluation:

```bash
uv run --frozen python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(
    'mlx-community/Nemotron-3-Diarization',
    revision='59ed2dbfc1346dcea9d423c71306a3a2499c568f',
    allow_patterns=['config.json', 'model.safetensors', 'README.md'],
))
PY
```

Retain model file hashes and its OpenMDW-1.1 license; this is a shared cache,
not a disposable run directory. Use a cached Sortformer snapshot only when it
is already available. The runner never downloads weights.

Optional public corpus preparation downloads individual AMI test meetings,
RTTM and UEM. It also builds controls and a constructed eight-identity case:

```bash
uv run --frozen python -m benchmarks.prepare_diarization_poc \
  --output "$EVIDENCE/public-corpus"
```

Read the AMI CC BY 4.0 terms linked in the report first. Retain the generated
`public-corpus.json`, source attribution and files. The source URLs/annotation
revision and content hashes are recorded; the audio mirror itself is not an
immutable Git revision. Preparation refuses to overwrite derived evidence.
Other recordings must be prepared as mono 16 kHz PCM16 WAV; preserve originals
and document the conversion recipe and hashes. Independent human RTTM and
optional UEM are required for accuracy claims.

## Guarded execution

Set `SERVICE_PID` to the **actual serving process**, whose children include any
resident inference worker, not just a shell or `uv` launcher. Verify its command,
creation time, working directory and health before running. Never stop the service
or unrelated processes to make the experiment fit.

```bash
uv run --frozen python -m benchmarks.diarization_poc \
  --model nemotron \
  --checkpoint /absolute/path/to/pinned/local/snapshot \
  --audio "$EVIDENCE/public-corpus/media/ES2004b.wav" \
  --reference "$EVIDENCE/public-corpus/references/ES2004b.rttm" \
  --recording-id ES2004b \
  --uem "$EVIDENCE/public-corpus/references/ES2004b.uem" \
  --protect-pid "$SERVICE_PID" \
  --output "$EVIDENCE/nemotron-ES2004b-r1.json"
```

One owned model process runs at a time. Offline variables are set in the child.
The supervisor checks production identity/children, a 20 GiB available-memory
reserve and an 8 GiB RSS cap at 250 ms intervals. A child monitor samples
MLX active + cached allocations and its reported peak every 50 ms, retaining
`.memory.json`, and terminates only itself if an observed value exceeds 8 GiB.
`set_memory_limit` is additionally configured but is only an allocator guideline.
Sampling can transiently overshoot; these are termination thresholds, not an
atomic memory reservation: a concurrent request
can begin between samples. SIGTERM is deferred during spawn, then the supervisor terminates and reaps its
owned process group through the same bounded cleanup path. The original signal
handler is restored after disposal. The deadline defaults to 900 seconds. Only the owned
process group is terminated if a guard fails; diagnostics remain on disk.

Use fresh output paths. JSON, log, NPZ probabilities, native RTTM and supervision
JSON are retained. An inference exception or supervisor abort is a failed run,
not a zero-error score. Completed native JSON/NPZ/RTTM are saved before reference
parsing and scoring. A scoring failure marks the JSON as `error` while preserving
those raw fields and sidecars; interim `inference_complete` is not a scored pass. Native out-of-range intervals produce `invalid_timeline`;
clipping for diagnostic scoring never passes the native output gate.

For a cropped reference input pass `--reference-offset SECONDS` in the original
recording's timebase. UEM and RTTM are cropped and shifted together. Non-finite, negative or reversed UEM coordinates are rejected before clipping.
A missing recording ID/reference region raises an error rather than silently scoring empty
reference speech. Reference-free silence has undefined DER; inspect predicted
speech duration and identities instead.

Nemotron uses defaults 0.5 / 0 / 0; Sortformer uses the historical adapter's
0.35 / 0.2 / 0.3. Override only on separate development material and freeze
before evaluation. Never compare a differently processed control as a model-only
ablation. Primary DER uses 0.25 s **total centered collar width**, overlap included,
and one recording-wide mapping. Zero collar is also saved.

For persistent-state PCM checks, use `--mode feed --feed-seconds 1`; the final
empty call flushes lookahead. For independent-file reset, add `--reset-check`
to a whole-file Nemotron run: A / one-second silence / A uses the same loaded
model and reports repeat probability/turn equality. Its timing is separate from
initial inference. Separate-process repetition alone does not test this state.

Timing synchronizes MLX work before stopping the clock. Decode/load/scoring/
artifact writing are excluded from inference RTF. Process RSS and MLX peak
allocations are separate measures; RSS alone can miss GPU memory pressure.
No runtime-memory estimate should be inferred from checkpoint file size.

## Tests and evidence

```bash
uv run --frozen pytest tests/unit/test_diarization_metrics.py tests/unit/test_diarization_poc.py
uv run --frozen mypy --no-incremental benchmarks/diarization_metrics.py \
  benchmarks/diarization_poc.py benchmarks/prepare_diarization_poc.py
```

The tests load no model. The interval scorer includes overlap, false alarms,
misses, global speaker swaps, duplicate same-speaker activity, collars and UEM
windows. It was also cross-checked against `pyannote.metrics==4.0.0` in a separate
reference-only uv environment; this package is not a project dependency.
Retain raw outputs, manifests, exact lock and scoring scripts before removing a
run environment. Do not commit private audio, local annotations or transcripts.
