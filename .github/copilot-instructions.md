# AI & ML Learning Repository - Copilot Instructions

Professional Python workspace for learning AI/ML concepts with structured module organization and test coverage.

## Project Architecture

**Core Philosophy**: Notebook-driven exploration → validated code migration → tested modules

- **`src/utils.py`** - Data processing utilities (loading, normalization, train-test splitting). All public functions must have docstrings and unit tests.
- **`notebooks/`** - Jupyter exploration ground; convert validated patterns to `src/` modules
- **`data/`** - Dataset storage (CSV files loaded via `load_data()` in utils)
- **`models/`** - Serialized model artifacts and checkpoints
- **`tests/test_utils.py`** - pytest suite; existing test uses pandas Series normalization as pattern

## Development Patterns & Conventions

### Data Processing Pattern (from `src/utils.py`)
- Import numpy/pandas at module top; use standard aliases (`np`, `pd`)
- Normalization: min-max scaling to [0,1] range; handle edge cases (data.min() == data.max())
- Train-test splitting: use fixed `random_state=42` for reproducibility; indices-based approach over sklearn
- All functions return pandas Series/DataFrames for consistency

### Testing Convention (from `tests/test_utils.py`)
- Import pytest and test functions; use descriptive docstrings
- Test files mirror source structure: `tests/test_*.py` for `src/*.py`
- Assert min/max bounds and data integrity, not just function execution
- Run locally: `pytest tests/ -v` (see pytest in requirements.txt)

## Critical Workflows

| Task | Command | Notes |
|------|---------|-------|
| Setup | `python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt` | Windows uses `venv\Scripts\activate` |
| Test | `pytest tests/ -v` | Run before committing; pytest v6.2.0+ in requirements |
| Format | `black src/` | Standard formatter (black v22.0.0+) |
| Lint | `flake8 src/` | Ensure PEP 8 compliance (flake8 v4.0.0+) |

## Key Dependencies & Versions
- **Data**: numpy ≥1.21, pandas ≥1.3 (Series/DataFrame API heavily used)
- **ML**: scikit-learn ≥1.0, scipy ≥1.7
- **DL**: tensorflow ≥2.8, torch ≥1.10, torchvision ≥0.11
- **Jupyter**: jupyterlab ≥3.0 (preferred over jupyter notebook)

## When Adding New Code

1. **Small utilities or tests**: Add to existing `src/utils.py` or `tests/test_utils.py`
2. **Separate modules**: Create `src/module_name.py` with docstrings and matching `tests/test_module_name.py`
3. **Experimental code**: Start in `notebooks/` (Jupyter cells), then migrate tested functions to `src/`
4. **Data files**: Place in `data/`, load via `load_data(filepath)` utility
5. **Models**: Save artifacts to `models/` directory after training

## Integration Points
- **GitHub**: Configured for remote push; use `git push -u origin main` for first push
- **CI/CD**: Ready for `.github/workflows/` GitHub Actions (pytest automation recommended)
- **External Data**: Use `load_data()` with optional preprocessing via `normalize_data()` or custom pipelines

---
Last Updated: January 24, 2026
