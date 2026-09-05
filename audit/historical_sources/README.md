# Recovered historical source contracts

These are archival source files, not replacement runtime modules. The current
`repro/2502.17749/v1/lpcode_v1/t3.py` remains unchanged.

| Archive | SHA-256 | Frozen binding |
| --- | --- | --- |
| t3_gate_a.py | 31a041fb71cf3064ffaf09fc3674a6914f64af21edc64eb58e9726c08f0107f5 | Gate A pair_builder_source_sha256 |
| t3_gate_b.py | 0d3ac0fcd714c98c3b674c63f25fbc267ad7a3b421cdf159ae52e6bbd1448ef5 | Gate B/C/D t3 implementation |

Recovered on 5 September 2026 from the project's recorded edit history.
The Gate B version was recovered by reversing the two recorded 4 September
negative-pair extension patches. The Gate A version was recovered by reversing
the subsequent Gate B runner additions and associated import/contract edits
recorded on 31 August UTC. The final files were accepted only after their
full byte-level SHA-256 values matched the pre-existing frozen configurations.
No history transcript, account metadata or unrelated user material is shipped.

The current module hash is
`ec7646e2671903c0cbd43f2854764eac6a93ee6dff7e04bc0a5eb2eaf5cf2060`.
It adds current/random/hard negative-pair selection; whole-file equality to
historical versions is therefore not claimed. Default-protocol equivalence
must be separately tested by raw-data pair reconstruction and ledger digest
comparison. Archive hash matches alone do not prove metric replication.

Do not copy these modules over the live package or resume old ledgers in
place. A historical retraining workflow requires an isolated version-matched
package and new output tree. All original copyright and licence exclusions
remain applicable; these files do not grant upstream LPcode rights.
