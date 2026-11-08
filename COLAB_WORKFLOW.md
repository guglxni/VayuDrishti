# Google Colab T4/L4 GPU Execution & Zindi Top-10 Code Review Workflow (`COLAB_WORKFLOW.md`)

**Competition:** Zindi × Google × Beyond × ClimateVerse — *Google Mumbai Air Quality Forecasting Challenge (2026)*  
**Primary Standalone Notebook:** `notebooks/Mumbai_Winning_Colab.ipynb`  
**CLI Orchestrator:** `scripts/colab_sync_and_run.py`  
**Verified Output Artifacts:** `submission.csv` and `submissions/submission_colab_verified.csv` (`2,592` rows)

---

## 1. Zindi Top-10 Code Review Compliance Checklist

Per `data/raw/CompetitionOverview.txt` (lines 150–160):
1. **"Custom packages in your submission notebook will not be accepted"**:
   - `notebooks/Mumbai_Winning_Colab.ipynb` has **zero local module imports** (`no import src.*`, `no import app.*`).
   - Every data loader, 2-stage sensor glitch + 7h Hampel outlier filter, 160-feature `TimeMixer` + $\tanh$-saturating CAMS MOS + `AirPhyNet` PDE pipeline, `TimesFM 3.0` (`V=4` & `V=12`), `Chronos-2` (`cross_learning=True`), 8-model zoo, 5-fold rolling CV evaluator, station-horizon variance shrinkage $\alpha_s(h_{\text{lead}})$, and 7-assertion submission verifier is implemented inline in 8 modular, self-contained cells.
2. **"Always set the seed. Rerunning your model should always place you at the same position on the leaderboard"**:
   - Cell 1 locks `SEED = 42`, `os.environ["PYTHONHASHSEED"] = "42"`, `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`, `torch.backends.cuda.matmul.allow_tf32 = False`, `torch.backends.cudnn.allow_tf32 = False`, and `torch.set_num_threads(1)`.
   - All GBDT learners enforce strict single-threaded deterministic execution:
     - `LightGBM`: `deterministic=True, force_col_wise=True, n_jobs=1, random_state=42`
     - `CatBoost`: `thread_count=1, random_seed=42`
     - `XGBoost`: `n_jobs=1, tree_method="hist", random_state=42`
3. **"You may only use publicly-available, open-source packages and pretrained models openly available to everyone"**:
   - Uses standard open-source PyPI packages (`timesfm[torch]==3.0.2`, `chronos-forecasting>=2.3.0`, `lightgbm`, `catboost`, `xgboost`, `scikit-learn`, `scipy`, `torch`, `huggingface_hub`) and open Hugging Face weights (`google/timesfm-3.0-pytorch`, `amazon/chronos-2`, `han-jisso/google-mumbai-air-quality-challenge`).

---

## 2. Step-by-Step Workflow: Staging Data to Hugging Face (`hf upload`)

To ensure zero-friction execution on a clean Google Colab runtime without manual file uploads, stage the competition dataset and pre-aligned external meteorological/CAMS tables to Hugging Face Hub:

### Option A: Using the `hf` / `huggingface-cli` CLI directly
```bash
# 1. Authenticate with Hugging Face Hub
hf auth login

# 2. Upload core competition files (Train.csv, Test.csv, Stations.csv, SampleSubmission.csv, AdditionalHistory.zip)
hf upload han-jisso/google-mumbai-air-quality-challenge data/raw/ . --repo-type dataset

# 3. Upload Open-Meteo Weather, Open-Meteo CAMS Air Quality, and VABB Airport METAR parquets
hf upload han-jisso/google-mumbai-air-quality-challenge data/external/ data/external --repo-type dataset

# 4. Upload cached foundation model & OOF validation artifacts for instant verification
hf upload han-jisso/google-mumbai-air-quality-challenge data/processed/ data/processed --repo-type dataset
```

### Option B: Using `scripts/colab_sync_and_run.py`
```bash
export HF_TOKEN="hf_your_write_token"
.venv/bin/python scripts/colab_sync_and_run.py --stage-hf --hf-repo han-jisso/google-mumbai-air-quality-challenge
```

---

## 3. Step-by-Step Workflow: Remote Execution on Google Colab T4/L4 GPU via `google-colab-cli`

You can execute `notebooks/Mumbai_Winning_Colab.ipynb` on a remote Google Colab `Linux x86_64 + NVIDIA T4/L4 GPU` directly from your terminal using `google-colab-cli`:

```bash
# 1. Authenticate google-colab-cli
colab auth login

# 2. Provision a remote Colab T4 GPU session
colab new --gpu T4 --name Mumbai_Winning_Colab

# 3. Execute notebooks/Mumbai_Winning_Colab.ipynb top-to-bottom on the remote T4 GPU
#    (Cell 1 auto-installs packages; Cell 2 pulls from Hugging Face Hub if data/ is not mounted)
colab exec notebooks/Mumbai_Winning_Colab.ipynb --output notebooks/Mumbai_Winning_Colab.ipynb

# 4. Download the generated submission.csv and verified submission artifact from the Colab VM
colab download submission.csv ./submission.csv
colab download submissions/submission_colab_verified.csv ./submissions/submission_colab_verified.csv
```

Or run the automated orchestrator wrapper in a single command:
```bash
.venv/bin/python scripts/colab_sync_and_run.py --colab-run --gpu T4
```

---

## 4. Step-by-Step Workflow: Local In-Place Notebook Execution & Verification

To execute `notebooks/Mumbai_Winning_Colab.ipynb` locally on macOS/Linux and embed all cell outputs, tables, and matplotlib charts directly inside the `.ipynb` file:

```bash
/Volumes/MacExt/Projects/google-mumbai-air-quality/.venv/bin/jupyter nbconvert \
  --to notebook --execute --inplace \
  notebooks/Mumbai_Winning_Colab.ipynb
```

Or via the helper script:
```bash
.venv/bin/python scripts/colab_sync_and_run.py --local-exec
```

---

## 5. Step-by-Step Workflow: Direct Zindi Submission via `zindi` SDK

Once `notebooks/Mumbai_Winning_Colab.ipynb` has generated and verified `submissions/submission_colab_verified.csv` (asserting `2,592` rows, exact `SampleSubmission.csv` `ID` alignment, finite, and non-negative values), push directly to the Zindi leaderboard:

```bash
export ZINDI_USERNAME="your_zindi_username"
export ZINDI_PASSWORD="your_zindi_password"
export ZINDI_CHALLENGE_ID="google-mumbai-air-quality-challenge"

.venv/bin/python scripts/colab_sync_and_run.py \
  --submit \
  --comment "8-Model Physics + TimesFM 3.0 (V=4/V=12) + Chronos-2 + GBDT + Wind-ST-GAT Ensemble (Colab Verified)"
```
