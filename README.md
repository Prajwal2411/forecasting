# HR Forecasting Starter

Minimal HR forecasting toolkit with a Streamlit-first UX, reproducible synthetic data, lightweight neural models, and an allocation optimizer. Upload CSVs or run the generators to get quarter-wise demand forecasts plus ranked internal candidate suggestions.

## Project Layout

- `app/streamlit_app.py` - Streamlit UI that assembles the JD parser, matcher, forecasting helpers, and optimizer from `src/` and writes the CSV summaries into `outputs/` and `data/` as needed.
- `app/file/data.py` - Faker-driven scenario builder that can produce ad-hoc employee, project, assignment, and market-trend CSVs for quick experiments.
- `scripts/generate_synthetic.py` - Deterministic CLI generator that writes reproducible employees/projects/assignments/market trends into `data/samples/`.
- `src/` - Core services:
   - `data_synthesis.py` boots up the `data/` directory with quarterly market trends plus helper functions for `gen_market_trends`, `gen_employees_projects_assignments`, and the shared `DATA_DIR`.
   - `jd_parser*.py`, `llm_match.py`, and `matching/embeddings.py` normalize requirements and match internal candidates with SentenceTransformer embeddings.
   - `forecasting.py`, `availability.py`, and `analytics_plan.py` orchestrate baseline forecasts, release notes, and plan-building helpers.
   - `optimizer.py` and `optimizer_lp.py` contain the greedy and ILP allocation solvers.
   - `models/train_mlp.py` trains the internal-fill classifier, `src/models/predictors.py` loads saved TensorFlow artifacts, and `src/utils/ontology.py` drives role/region metadata plus `hiring_rule.yml` and `roles.yml`.
- `data/` - Default data store for `employees.csv`, `projects.csv`, `assignments.csv`, and `market_trends_quarterly.csv`. The subfolder `data/samples/` holds the generator output.
- `models/registry/` - Saved TensorFlow models plus preprocessing artifacts and metadata (update `meta.json` after retraining).
- `outputs/` - `forecasts.csv` and `release_suggestions.csv` produced by the optimizer.
- `hiring_rule.yml`, `roles.yml`, `sample_jd.txt` - Configuration files referenced by the parser and UI.

## Quickstart

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

1. Regenerate sample data (writes to `data/samples/`).
   ```bash
   python scripts/generate_synthetic.py --seed 42
   ```
2. Train or refresh the internal-fill classifier/regressor.
   ```bash
   python -m src.models.train_mlp --data-dir data/samples --model-dir models/registry/v1
   ```
3. Launch the Streamlit experience.
   ```bash
   streamlit run app/streamlit_app.py
   ```

The Streamlit app will also regenerate the `data/` CSVs via `src/data_synthesis.py` if they are missing or outdated.

## Data Flow

1. Upload or reuse the four required CSVs (`projects.csv`, `assignments.csv`, `employees.csv`, `market_trends_quarterly.csv`). The schema validator enforces structure + datatypes.
2. Build a staffing request (currency, budget, quarters, team size, JD, tech stack, profit margin).
3. The JD parser normalizes skills, infers seniority/urgency, and the embedding matcher scores internal candidates.
4. Cached SentenceTransformer embeddings and TensorFlow MLPs estimate internal fill probability and time-to-fill for each quarter.
5. Baseline demand forecast aggregates market & assignment history per quarter/role.
6. The greedy or PuLP ILP optimizer splits demand across internal vs external hires with budget + capacity constraints and surfaces release suggestions.
7. Forecast and release CSVs land in `outputs/` and are downloadable from the UI.

## Environment

- Uses `python-dotenv` to read optional `.env` values (LLMs, embedding overrides, etc.).
- Sentence-transformer default: `sentence-transformers/all-MiniLM-L6-v2`.
- TensorFlow models remain small (two dense layers) so CPU-only training stays under a minute.
- `app/streamlit_app.py` caches embeddings and models to avoid re-loading between reruns.

## Notes

- Update `models/registry/meta.json` when you retrain so the UI can surface training provenance.
- The fast data pathways under `app/file/data.py` and `scripts/` are deterministic when reusing the same `--seed`.
