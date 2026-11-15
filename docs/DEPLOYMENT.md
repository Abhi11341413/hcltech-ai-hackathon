# Deploying the public demo

The demo is the Streamlit UI (`src/ui/app.py`), which runs the whole pipeline in one process.

| Host | Cost | How |
|---|---|---|
| **Streamlit Community Cloud** (used) | Free | Deploys straight from the GitHub repo, see below |
| Hugging Face Spaces (Docker) | Needs a PRO subscription for Docker Spaces | `Dockerfile` + `scripts/deploy_hf_space.py`, see further below |
| Any Docker host (Cloud Run, a VM, ...) | Varies | `Dockerfile` |

## Streamlit Community Cloud (free)

`requirements.txt` pulls the **CPU-only** PyTorch build (`--extra-index-url .../whl/cpu`), so the
install is ~200 MB instead of ~2.5 GB of CUDA libraries that Streamlit Cloud cannot use.

1. Sign in at https://share.streamlit.io with the GitHub account that can see this repository
   and allow Streamlit to access it (needed because the repository is private).
2. **Create app** → *Deploy a public app from GitHub* and fill in:

   | Field | Value |
   |---|---|
   | Repository | `Abhi11341413/hcltech-ai-hackathon` |
   | Branch | `main` (or a feature branch to try it before merging) |
   | Main file path | `src/ui/app.py` |
   | App URL | pick a name, e.g. `enterprise-doc-qa` |

3. **Advanced settings** → Python version **3.11** (or 3.12), and paste the key into **Secrets**
   (TOML; top-level secrets are exposed as environment variables, which the LLM client reads):

   ```toml
   GEMINI_API_KEY = "..."      # or ANTHROPIC_API_KEY = "..."
   ```

4. **Deploy**. The first build takes ~5 minutes (installing packages); the embedding model is
   downloaded on the first "Load sample documents", which takes about a minute.

Every push to the chosen branch redeploys automatically. Logs: the app's **Manage app** panel.
Free apps sleep after a period without visitors; opening the URL wakes them.

## Hugging Face Spaces (Docker)

Hugging Face now requires a PRO subscription for Docker Spaces on free CPU hardware. If the team has
one, this gives 16 GB RAM and the image below, with the embedding model baked in.

| File | Purpose |
|---|---|
| `Dockerfile` | Python 3.11, CPU-only PyTorch, project requirements, embedding model downloaded at build time, Streamlit on port 7860 |
| `.dockerignore` | Keeps `.env`, `data/uploads`, `models`, caches and tests out of the image |
| `scripts/deploy_hf_space.py` | Uploads the git-tracked files to the Space, adding the Space's README front matter on the fly |

### 1. Test the container locally (needs Docker Desktop running)

```bash
docker build -t docqa-demo .
docker run --rm -p 7860:7860 docqa-demo                      # offline extractive answers
docker run --rm -p 7860:7860 -e GEMINI_API_KEY=... docqa-demo  # or -e ANTHROPIC_API_KEY=...
```

Open http://localhost:7860, load the sample documents in the **Knowledge base** tab and ask a question.
Never put a key in the Dockerfile or commit it; pass it with `-e` only.

### 2. One-time Hugging Face setup (each person deploying)

1. Create a free account at https://huggingface.co/join.
2. Create an access token with **Write** permission: https://huggingface.co/settings/tokens.
3. Log in from the project folder and paste the token when asked (it is stored in your user
   profile, not in the repo):

   ```bash
   hf auth login
   ```

### 3. Deploy

Commit your changes first (only git-tracked files are uploaded), then:

```bash
python -m scripts.deploy_hf_space --space <hf-username>/enterprise-doc-qa
```

The script creates the Space if needed and uploads the files. Hugging Face then builds the
image (about 5–10 minutes the first time); follow it in the Space's **Logs** tab. The app is at
`https://huggingface.co/spaces/<hf-username>/enterprise-doc-qa`.

Use `--dry-run` to list what would be uploaded, `--private` for a private Space.
Re-run the same command to deploy a newer commit.

### 4. Add the LLM key as a secret

Space → **Settings** → **Variables and secrets** → **New secret**:

| Name | Value |
|---|---|
| `GEMINI_API_KEY` *or* `ANTHROPIC_API_KEY` | your key |

The Space restarts automatically. Without a key the app still works and answers with the
offline extractive fallback (it quotes the best passages instead of writing a summary).
Optional non-secret variables use the normal overrides, e.g. `QA_GENERATION__PROVIDER=gemini`.

## Before sharing the link publicly

* **Cost:** every question calls the LLM on your key. Set a spending limit in the provider's
  console (Google AI Studio / Anthropic Console).
* **Uploads:** visitors can upload documents. They are stored in the container's temporary disk
  and disappear on restart, but anyone can see answers built from them during that time.
  Do not upload confidential documents to the public Space.
* **Prompt injection:** uploaded text is untrusted. The generation prompt instructs the model
  to ignore instructions inside passages (CLAUDE.md §10), and claim verification removes
  unsupported claims.
* **Sleeping:** free Spaces sleep after ~48 h without visitors; the first request wakes them
  (about a minute).
* **XSRF:** the Dockerfile disables Streamlit's XSRF check because Spaces serves the app in a
  cross-origin iframe, where the check blocks file uploads.
