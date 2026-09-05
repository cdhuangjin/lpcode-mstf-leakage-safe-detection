# CodeBERT feasibility — no training
Date: 2026-09-05. RECOMMEND_RUN_CODEBERT = **NO for this round**.

The [official implementation](https://github.com/microsoft/CodeBERT) uses a RoBERTa-based encoder. Its [published configuration](https://huggingface.co/microsoft/codebert-base/raw/main/config.json) has 12 layers, hidden size 768 and 514 position embeddings. C/C++ are outside the six pretraining languages listed by the authors; this is a coverage limitation, not proof of poor performance.

1. Pair input: encode source and candidate jointly using the checkpoint tokenizer's pair interface and special tokens; attach a supervised binary head. The task is not single-candidate classification. Confirm the emitted token sequence in a smoke test instead of assuming literal BERT [CLS]/[SEP] IDs.
2. Truncation: plan a 512-token input budget including special tokens, subject to pinned tokenizer verification. The proportion of truncated pairs has **not been measured**. Before training, measure per-endpoint and combined token-length distributions on training data and freeze the truncation policy without test-score selection. Do not claim truncation is mild or severe yet.
3. Hardware: a read-only nvidia-smi snapshot showed RTX 5060, 8,151 MiB total and 2,797 MiB used. This is not a memory reservation or a demonstrated fitting capacity. Batch size, precision, gradient accumulation and sequence lengths require an isolated pilot. Other processes were not interrupted.
4. Training time and peak memory: unmeasured. No checkpoint, tokenizer or training dependencies were downloaded for this assessment; no benchmark duration is invented.
5. Fair matching: possible in principle for existing LPcode manifests if identical labels, split/pair hashes, training-only model selection and seeds are used. CodeMirage matching is presently blocked by the same missing provenance.
6. Priority: resolve external task eligibility before a new neural baseline. A later separately approved LPcode pilot could address classifier breadth, but would not repair missing external provenance. Pin model/tokenizer revisions and dependencies in a separate environment first.
