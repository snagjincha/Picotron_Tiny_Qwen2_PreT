# Pretraining Practice

Qwen2-style dense language-model pretraining experiments, starting with a
100M-parameter model trained on a deterministic, cleaned C4-English subset.

## 1. 100M experiment

The initial model keeps the Tiny Qwen2 dense design while using the original
Qwen2 tokenizer vocabulary (151,936 tokens). Tied embeddings make the model
approximately 100.8M parameters rather than 30M.

- Architecture: 8 layers, hidden size 512, 8 attention heads, 4 KV heads,
  RoPE, RMSNorm, SwiGLU, and tied input/output embeddings.
- Data: C4 English read in streaming mode, cleaned and split by document.
- Compute-optimal budget: 2,016,061,440 packed train tokens (20 tokens per
  parameter).
- Training framework: Picotron, initially with data parallelism only.
- Optimizers: all-parameter AdamW baseline or Muon hybrid (Muon for Transformer
  matrices and auxiliary AdamW for embedding, tied head, and RMSNorm).

## 2. Environment

```bash
cd /mnt/raid5/chasj/Research/Pretraining_Practice
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you already use conda on the server, you can use a conda environment instead.

Picotron is intentionally kept as a separately pinned checkout under
`third_party/picotron/`; its exact commit will be recorded before the first
distributed run.

## 3. Verify the 100M configuration

```bash
python scripts/verify_model_config.py
```

This checks the architecture-derived parameter count and computes the 20:1
Chinchilla token budget. It does not download a model or dataset.

## 4. Model unit tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The tests cover logits shape, causal masking, tied embeddings, and configured
context length enforcement.

## 5. Mandatory overfit test

```bash
PYTHONPATH=src python3 scripts/run_overfit_test.py --steps 80
```

This fits one fixed next-token batch with the selected optimizer. The command
fails unless its loss falls by at least 50%; run it before any C4 smoke test.

Once C4 tokens are prepared, a single-process smoke run is available with:

```bash
PYTHONPATH=src python3 scripts/train.py \
  --train-tokens data/tokenized/qwen2_100m_c4_en/train.bin \
  --validation-tokens data/tokenized/qwen2_100m_c4_en/validation.bin \
  --max-steps 50
```

Picotron replaces this single-process launcher only after the same smoke
workflow is validated.

## 6. Prepare C4 token streams

```bash
PYTHONPATH=src python3 scripts/prepare_c4.py
```

This streams C4 English, performs deterministic cleaning/deduplication and a
document-level 99:1 split, then writes `uint32` packed token streams and a
manifest under `data/tokenized/qwen2_100m_c4_en/`. The configured 2B-token
run needs approximately 8 GB for the train token stream alone; run it only on
the intended storage volume.

## 7. Existing tokenizer/model loading smoke test

This verifies that both tokenizer and model can be loaded.

```bash
python scripts/load_model_tokenizer.py
```

Default model:

```text
sshleifer/tiny-gpt2
```

You can choose another small causal language model:

```bash
python scripts/load_model_tokenizer.py --model-name distilgpt2
```

## 8. Sync Back From Local Codex Workspace

From the local Codex workspace:

```bash
rsync -avz ./Pretraining_Practice/ chasj@10.0.12.120:/mnt/raid5/chasj/Research/Pretraining_Practice/
```
