# Deployment

| file | |
|---|---|
| `MODEL_CARD.md` | uploaded as the Hugging Face repo `README.md` |
| `upload_to_hf.py` | pushes the GGUF, sidecar and card in one commit |

```sh
python upload_to_hf.py --dry-run    # validate, print the plan, send nothing
python upload_to_hf.py --confirm    # publish
```

Uploading is public and effectively irreversible — a pushed commit stays in the
history after deletion, and weights get mirrored quickly. The script therefore
uploads nothing without `--confirm`, and validates first: GGUF magic bytes, file
size within 2% of the expected 3.48 GB, and the card present. All three files go
in a single commit so the repo is never in a state where the weights are public
but the card documenting the required `repeat_last_n=2048` is not.

Auth via `huggingface-cli login` or `$HF_TOKEN`.

**Note:** the target namespace was specified as both `sgsystems` and
`SG-Systems`. HF repo ids are case-sensitive; the default is
`sgsystems/Falcon-H1-7B-FORGE-v2`, override with `--repo`.
