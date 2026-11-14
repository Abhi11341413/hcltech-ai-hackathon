# Team Git workflow

**On your laptop:** clone, switch branch, edit, test, `status`, `add`, `commit`, `push`, `pull`, resolve conflicts.
**On GitHub (website or `gh`):** create Pull Requests, review, approve, merge, invite collaborators.

Repository: `https://github.com/Abhi11341413/hcltech-ai-hackathon` (private).

| Member | Branch |
|---|---|
| Member 1 — Team Lead | `feature/member-1-reasoning-evaluation` |
| Member 2 | `feature/member-2-ingestion-api` |
| Member 3 | `feature/member-3-hybrid-retrieval` |
| Member 4 | `feature/member-4-grounded-generation` |

## 0. One-time: Lead invites collaborators (GitHub)

```bash
gh api -X PUT repos/Abhi11341413/hcltech-ai-hackathon/collaborators/<github-username> -f permission=push
```
Or in the browser: repo → Settings → Collaborators → Add people. Each member accepts the email invite.

## 1–3. Clone, switch to your branch, check it (laptop)

```bash
git clone https://github.com/Abhi11341413/hcltech-ai-hackathon.git
cd hcltech-ai-hackathon
python -m venv .venv && .venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

| Member | Switch command |
|---|---|
| 1 | `git switch feature/member-1-reasoning-evaluation` |
| 2 | `git switch feature/member-2-ingestion-api` |
| 3 | `git switch feature/member-3-hybrid-retrieval` |
| 4 | `git switch feature/member-4-grounded-generation` |

```bash
git branch --show-current     # must print YOUR branch before you edit anything
```

## 4–8. Edit, check, stage, commit, push (laptop)

Example for **Member 3** (others: replace files and branch):

```bash
# edit src/retrieval/hybrid.py, remove xfail markers in tests/unit/test_retrieval.py for implemented functions
python -m pytest tests/unit/test_retrieval.py -q
git status                                   # see what changed
git add src/retrieval/hybrid.py tests/unit/test_retrieval.py   # stage only YOUR files (avoid "git add .")
git commit -m "feat(retrieval): implement reciprocal rank fusion"
git push -u origin feature/member-3-hybrid-retrieval           # -u only the first time; afterwards: git push
```

Typical commits per member:

* Member 1: `git add src/reasoning/ tests/unit/test_reasoning_and_eval.py` → `git commit -m "feat(reasoning): detect value and version conflicts"`
* Member 2: `git add src/ingestion/ tests/unit/test_ingestion.py` → `git commit -m "feat(ingestion): markdown and html loaders with heading paths"`
* Member 3: `git add src/retrieval/ config/glossary.yaml` → `git commit -m "feat(retrieval): bm25 + dense hybrid retriever"`
* Member 4: `git add src/generation/ tests/unit/test_generation.py` → `git commit -m "feat(generation): structured grounded answers with citations"`

## 9. Create a Pull Request (GitHub)

```bash
gh pr create --base main --head feature/member-3-hybrid-retrieval \
  --title "feat(retrieval): hybrid BM25 + dense retrieval" \
  --body "Implements FR-03. Test evidence: <paste pytest output>"
```
Or on GitHub: "Compare & pull request". The PR template asks for test evidence and a contract-change checklist.

## 10. Review and merge (GitHub)

Reviewer (a different member):
```bash
gh pr list
gh pr checkout <number>          # test it locally
python -m pytest -q
gh pr review <number> --approve  # or --request-changes -b "reason"
```
Author or Lead, after approval:
```bash
gh pr merge <number> --squash --delete-branch=false
```
Keep the feature branch if more work follows; otherwise delete it.

## 11. Bring the latest main into your feature branch (laptop)

```bash
git switch feature/member-2-ingestion-api
git fetch origin
git merge origin/main             # or: git pull origin main
python -m pytest -q
git push
```

## 12. Resolving a merge conflict (laptop)

```bash
git merge origin/main
# CONFLICT (content): Merge conflict in src/schemas.py
git status                        # lists conflicted files
# open the file, find <<<<<<< ======= >>>>>>> markers, keep the correct combined version, delete markers
python -m pytest -q
git add src/schemas.py
git commit                        # completes the merge
git push
```
If the conflict is in a file you do not own: stop and talk to its owner before choosing a side.
To abort: `git merge --abort`.

## 13. Integration tests after merging (laptop, on main)

```bash
git switch main
git pull origin main
python -m pytest -q
python -m scripts.mock_e2e
python -m src.evaluation.run_eval      # once Member 1's runner is merged
uvicorn src.api.app:app --reload       # manual smoke test at /docs
```

## Integration rules

* `main` always runnable; no direct commits to main; every module via PR with description + test evidence.
* Another member reviews every PR. Merge small pieces often (every ~60–90 min).
* Run integration tests after every important merge.
* Changes to `src/schemas.py`, `src/errors.py`, `src/config.py`, `src/mocks.py` need all four to agree.

## Integration order

1. **Skeleton + contracts** (done on main).
2. **M2 ingestion/chunking** → real chunks for everyone.
3. **M3 retrieval** → depends on Chunk only (develop against `mocks.mock_chunks()` before step 2 lands).
4. **M4 generation** → depends on RetrievedPassage only (`mocks.mock_passages()`).
5. **M1 reasoning + pipeline** → wires 2–4 together (uses mocks until they land).
6. **M2 API endpoints** call `QAPipeline`.
7. **M1 evaluation + report**, M3 ablation, everyone adds eval questions and demo examples.

## Recommended GitHub setting (Lead)

Settings → Branches → Add rule for `main`: require a pull request before merging, require 1 approval.
(Branch protection on private repos needs GitHub Pro/Team; otherwise follow the rule by discipline.)
