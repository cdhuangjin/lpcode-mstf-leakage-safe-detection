# Feature-importance mechanism analysis

Frozen registry SHA-256: `f18c06f33f3f7f4037c074c58543ed7ff96342108bb4d39838262b626974df62`.

This is descriptive: importance is computed from fixed XGBoost fits on reconstructed saved splits, and does not establish a causal feature effect.

- `clean`: highest grouped held-out permutation decrease is `relative_delta:structural_syntax` (0.008462; 60 reconstructed folds).
- `combined_attack`: highest grouped held-out permutation decrease is `relative_delta:structural_syntax` (0.024320; 60 reconstructed folds).
- `cross_language`: highest grouped held-out permutation decrease is `delta:original_style` (0.009455; 12 reconstructed folds).
- `unseen_llm`: highest grouped held-out permutation decrease is `relative_delta:structural_syntax` (0.010952; 240 reconstructed folds).
