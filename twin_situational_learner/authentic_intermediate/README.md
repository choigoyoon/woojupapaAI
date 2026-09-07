# Authentic intermediate ledger upload target

Copy the original files from `E:\Twin\scratch\learning_audit_logs` into this
directory on the source PC.  Parquet files in this directory are tracked by
Git LFS because several exceed GitHub's normal file-size limit.

The first exact-parity pass only needs Stages 09-11.  Stages 03-08 are needed
afterward to prove the full reverse handoff with the recorded source hashes.

| Stage | File | Bytes | Required SHA-256 |
|---:|---|---:|---|
| 03 | `module_03_candidate_patterns.parquet` | 8,687,023 | `863e80f4ed23ffeace8065708df2343137c00cc0405b6e29b4da35939fc87733` |
| 04 | `module_04_context_patterns.parquet` | 8,815,556 | `7b98757f5cbddc1927f55b38b698f720730cfbb758259ea5b9a24684167e4352` |
| 05 | `module_05_time_patterns.parquet` | 13,812,742 | `63a9bfbca375920b02a22e0d92a841086d0ed94c4659d83ab4822d81e800421b` |
| 06 | `module_06_candle_volume_patterns.parquet` | 44,854,346 | `901c73bebb1dcdb0c0d044f7570ce5738a5b319f56a9a9207a586dd6be71970a` |
| 07 | `module_07_multitf_macd_patterns.parquet` | 138,590,917 | `43560fea7170736ad030ef148a94d027f47ae383a73dd4bac17c89eadfc094a8` |
| 08 | `module_08_flow_order_patterns.parquet` | 139,458,085 | `c3d1632100c6d0da68ad395e54d52aba991187dd8919c24dbec2ed9f2ea4e4ad` |
| 09 | `module_09_rule_recognition_patterns.parquet` | 170,837,594 | `2b9ea0fb79887fb9ed4a5b017455d758769f4bf3c396c46efeb25c2941a4ac98` |
| 10 | `module_10_wait_reason_patterns.parquet` | 173,709,044 | `dca1ea2a7710ec64388f9154dc35aa6ac6d8b1e6d96adeb13418377df1ca554b` |
| 11 | `module_11_final_action_patterns.parquet` | 174,055,182 | `de03c08d427aa215a4c79ca5ecd444ced09a0298fe542d5604720d81f5ec0ce5` |

Do not rename the files.  The combined recorded size is 872,820,489 bytes.
