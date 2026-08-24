# Third-party model notices

This container redistributes the following pinned model artifacts. Their Hugging
Face model cards declare the Apache License 2.0:

| Artifact | Revision | Copyright / author information supplied upstream |
| --- | --- | --- |
| [`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286` | SpeechBrain project contributors |
| [`griko/gender_cls_svm_ecapa_voxceleb`](https://huggingface.co/griko/gender_cls_svm_ecapa_voxceleb) | `25f3e5a3c1c172dceeb723d8061e3e80ba6c8d64` | Upstream model author `griko` |
| [`griko/age_reg_svr_ecapa_voxceleb2`](https://huggingface.co/griko/age_reg_svr_ecapa_voxceleb2) | `1d2356ac55f51fbd3f327f1b9260860decb21233` | Upstream model author `griko` |

The complete Apache License 2.0 text is included in the image at
`/licenses/Apache-2.0.txt`. The service's MIT license is included at
`/licenses/service-MIT.txt`. Exact model file hashes and immutable sources are in
`/opt/models/model-metadata.json` and `backend/scripts/prepare_models.py`.

The upstream repositories did not expose separate `LICENSE` files at the pinned
model revisions; their model-card license declarations are linked and discussed
in `backend/docs/ADR-001-model-selection.md`. No endorsement by the model authors is
implied. Model and training-data terms are distinct from this service's code
license, and downstream distributors remain responsible for legal review and any
required attribution or notice updates.
