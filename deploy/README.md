# Deployment

| file | |
|---|---|
| `MODEL_CARD.md` | uploaded as `sgsystems/Falcon-H1-7B-FORGE-v2/README.md` |
| `ORG_CARD.md` | uploaded as the **Space** `sgsystems/README` — the org profile at huggingface.co/sgsystems |
| `upload_to_hf.py` | pushes the GGUF, sidecar and card in one commit |

**The org card is a static Space containing only `README.md`.** Two things have
to be right, and getting either wrong fails in a differently confusing way:

1. **It must be a Space, not a model repo.** A model repo named `README` leaves
   the profile showing "No organization card" *and* appears in the org's model
   list, so one model looks like two.

2. **`index.html` must not exist.** `create_repo(space_sdk="static")` scaffolds
   `index.html` and `style.css`, and a static Space serves `index.html` — so the
   profile renders "Welcome to your static Space!" instead of the card. Working
   org cards (`google/README`, `meta-llama/README`, `huggingface/README`) all
   ship exactly `.gitattributes` + `README.md`.

```python
api.create_repo("<org>/README", repo_type="space", space_sdk="static", exist_ok=True)
api.create_commit(repo_id="<org>/README", repo_type="space",
                  operations=[CommitOperationAdd("README.md", "ORG_CARD.md")])
# then delete the scaffold, or the card will not render:
api.create_commit(repo_id="<org>/README", repo_type="space",
                  operations=[CommitOperationDelete("index.html"),
                              CommitOperationDelete("style.css")])
```

Frontmatter is Space metadata (`title`, `emoji`, `colorFrom`/`colorTo`,
`sdk: static`, `pinned`), not a document title.

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

**Namespace.** HF repo ids are case-sensitive. `SG-Systems` returns 404; the
org's canonical id is `sgsystems` (full name "SG Systems"), which is the
default. Override with `--repo`.

**Token scope.** Writing to an org needs a token scoped to *that org*, not just
to your user. A fine-grained token scoped only to `user:<you>` authenticates
fine and then fails at `create_repo` with a 403 — a poor way to find out
partway through a 3.5 GB upload. Verify first:

```sh
python -c "from huggingface_hub import HfApi; w=HfApi().whoami(); \
print([e['entity']['name'] for e in \
w['auth']['accessToken'].get('fineGrained',{}).get('scoped',[]) \
if e['entity']['type']=='org'])"
```
