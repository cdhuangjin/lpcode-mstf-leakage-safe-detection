# Manuscript evidence audit

Status: PASS

Gate A-D, A0-A5 means, negative-pair means and paired deltas; CIs not recomputed

Arithmetic uses ledger F1 cells only. Expected claims are validation targets.
Each comparison enforces the full registered Cartesian cell set and exact train/test hashes.
F1 tolerance: 0.0001; delta tolerance: 0.001 pp. Named manuscript cells must match exact rounding.

| Check | Computed | Expected | Observed | Status |
| --- | --- | --- | --- | --- |
| A: source hash | 7c2c7887a0ee795b05d633eb3ddf2f4bc7fc9f27f503e5e9c8956ca5761157bd | 7c2c7887a0ee795b05d633eb3ddf2f4bc7fc9f27f503e5e9c8956ca5761157bd | 7c2c7887a0ee795b05d633eb3ddf2f4bc7fc9f27f503e5e9c8956ca5761157bd | PASS |
| B: source hash | ee56023c4e4555e7a826708f9da7f279cab4eb06fdebbde2adcfc69997a57fae | ee56023c4e4555e7a826708f9da7f279cab4eb06fdebbde2adcfc69997a57fae | ee56023c4e4555e7a826708f9da7f279cab4eb06fdebbde2adcfc69997a57fae | PASS |
| C: source hash | fddc0c61fea7dfdd2658d6e3b5119b63ae8a2c604606dfb3d1e8c6ddae065c0b | fddc0c61fea7dfdd2658d6e3b5119b63ae8a2c604606dfb3d1e8c6ddae065c0b | fddc0c61fea7dfdd2658d6e3b5119b63ae8a2c604606dfb3d1e8c6ddae065c0b | PASS |
| D: source hash | ea714f618542ece8d101d24a6edd558b91c362c3916bafb3351ce5e99283ffe6 | ea714f618542ece8d101d24a6edd558b91c362c3916bafb3351ce5e99283ffe6 | ea714f618542ece8d101d24a6edd558b91c362c3916bafb3351ce5e99283ffe6 | PASS |
| ablation: source hash | 73c24ae80ab6dcfa36acb857456818e005cbfbedc2197fb720c0098d7f300903 | 73c24ae80ab6dcfa36acb857456818e005cbfbedc2197fb720c0098d7f300903 | 73c24ae80ab6dcfa36acb857456818e005cbfbedc2197fb720c0098d7f300903 | PASS |
| negative: source hash | 7aebf034cdd3011d1d24def8640b04c19638607080894145197a95fb9498159b | 7aebf034cdd3011d1d24def8640b04c19638607080894145197a95fb9498159b | 7aebf034cdd3011d1d24def8640b04c19638607080894145197a95fb9498159b | PASS |
| Gate A: complete cells and matched train/test hashes | 60 | 60 | 60 | PASS |
| Gate A: baseline | 0.9149157323442745 | 0.9149 | 0.9149 | PASS |
| Gate A: method | 0.9317469169035462 | 0.9317 | 0.9317 | PASS |
| Gate A: delta_pp | 1.6831184559271604 | 1.683 | +1.683 | PASS |
| Gate B: complete cells and matched train/test hashes | 240 | 240 | 240 | PASS |
| Gate B: baseline | 0.8914988430842783 | 0.8915 | 0.8915 | PASS |
| Gate B: method | 0.9671298032333194 | 0.9671 | 0.9671 | PASS |
| Gate B: delta_pp | 7.563096014904112 | 7.563 | +7.563 | PASS |
| Gate C: complete cells and matched train/test hashes | 60 | 60 | 60 | PASS |
| Gate C: baseline | 0.8415260387132605 | 0.8415 | 0.8415 | PASS |
| Gate C: method | 0.9531278544597213 | 0.9531 | 0.9531 | PASS |
| Gate C: delta_pp | 11.160181574646076 | 11.16 | +11.160 | PASS |
| Gate D: complete cells and matched train/test hashes | 12 | 12 | 12 | PASS |
| Gate D: baseline | 0.8938137486214468 | 0.8938 | 0.8938 | PASS |
| Gate D: method | 0.9652215804841137 | 0.9652 | 0.9652 | PASS |
| Gate D: delta_pp | 7.140783186266681 | 7.141 | +7.141 | PASS |
| A0 clean: complete cells and matched train/test hashes | 60 | 60 | 60 | PASS |
| A0 clean: F1 | 0.9149157323442745 | 0.9149 | 0.9149 | PASS |
| A0 unseen: complete cells and matched train/test hashes | 240 | 240 | 240 | PASS |
| A0 unseen: F1 | 0.9089372771555689 | 0.9089 | 0.9089 | PASS |
| A1 clean: complete cells and matched train/test hashes | 60 | 60 | 60 | PASS |
| A1 clean: F1 | 0.9317469169035462 | 0.9317 | 0.9317 | PASS |
| A1 unseen: complete cells and matched train/test hashes | 240 | 240 | 240 | PASS |
| A1 unseen: F1 | 0.9204700419564912 | 0.9205 | 0.9205 | PASS |
| A2 clean: complete cells and matched train/test hashes | 60 | 60 | 60 | PASS |
| A2 clean: F1 | 0.9608392760430582 | 0.9608 | 0.9608 | PASS |
| A2 unseen: complete cells and matched train/test hashes | 240 | 240 | 240 | PASS |
| A2 unseen: F1 | 0.9586926313377425 | 0.9587 | 0.9587 | PASS |
| A3 clean: complete cells and matched train/test hashes | 60 | 60 | 60 | PASS |
| A3 clean: F1 | 0.9682945361059245 | 0.9683 | 0.9683 | PASS |
| A3 unseen: complete cells and matched train/test hashes | 240 | 240 | 240 | PASS |
| A3 unseen: F1 | 0.964108952932598 | 0.9641 | 0.9641 | PASS |
| A4 clean: complete cells and matched train/test hashes | 60 | 60 | 60 | PASS |
| A4 clean: F1 | 0.9726656538111014 | 0.9727 | 0.9727 | PASS |
| A4 unseen: complete cells and matched train/test hashes | 240 | 240 | 240 | PASS |
| A4 unseen: F1 | 0.967057722913079 | 0.9671 | 0.9671 | PASS |
| A5 clean: complete cells and matched train/test hashes | 60 | 60 | 60 | PASS |
| A5 clean: F1 | 0.9724806385938991 | 0.9725 | 0.9725 | PASS |
| A5 unseen: complete cells and matched train/test hashes | 240 | 240 | 240 | PASS |
| A5 unseen: F1 | 0.9671298032333194 | 0.9671 | 0.9671 | PASS |
| negative current: complete cells and matched train/test hashes | 60 | 60 | 60 | PASS |
| negative current: baseline | 0.9149157323442745 | 0.9149 | 0.9149 | PASS |
| negative current: method | 0.9317469169035462 | 0.9317 | 0.9317 | PASS |
| negative current: delta_pp | 1.6831184559271604 | 1.683 | +1.683 | PASS |
| negative random: complete cells and matched train/test hashes | 60 | 60 | 60 | PASS |
| negative random: baseline | 0.9119408464963992 | 0.9119 | 0.9119 | PASS |
| negative random: method | 0.9299871418711275 | 0.93 | 0.9300 | PASS |
| negative random: delta_pp | 1.8046295374728367 | 1.805 | +1.805 | PASS |
| negative hard: complete cells and matched train/test hashes | 60 | 60 | 60 | PASS |
| negative hard: baseline | 0.8463202413433392 | 0.8463 | 0.8463 | PASS |
| negative hard: method | 0.882370405486086 | 0.8824 | 0.8824 | PASS |
| negative hard: delta_pp | 3.605016414274688 | 3.605 | +3.605 | PASS |

## Source SHA-256 hashes

- `FILE_MANIFEST.json`: `09e4a39142d62574235885d966709f39ac959ea9d3400d0c624fde67601406d0`
- `results/01_transition_test_strict_origins/folds.jsonl`: `7c2c7887a0ee795b05d633eb3ddf2f4bc7fc9f27f503e5e9c8956ca5761157bd`
- `results/02_unseen_llm/folds.jsonl`: `ee56023c4e4555e7a826708f9da7f279cab4eb06fdebbde2adcfc69997a57fae`
- `results/03_style_attack/folds.jsonl`: `fddc0c61fea7dfdd2658d6e3b5119b63ae8a2c604606dfb3d1e8c6ddae065c0b`
- `repro/2502.17749/v1/results/04_cross_language/folds.jsonl`: `ea714f618542ece8d101d24a6edd558b91c362c3916bafb3351ce5e99283ffe6`
- `results/05_mechanism_analysis/folds.jsonl`: `73c24ae80ab6dcfa36acb857456818e005cbfbedc2197fb720c0098d7f300903`
- `results/negative_pair_robustness/raw_results.json`: `7aebf034cdd3011d1d24def8640b04c19638607080894145197a95fb9498159b`
- `results/07_manuscript/paper_revised.md`: `4a6828f7d0e8982b2184180b98eeae0cc59b33be196d1e2193cb3e91a4a6bfaa`
