Project Environment & Notebook Setup
=================================

Summary
-------
- Consolidated Python virtual environments and standardized on a single `.venv` created with Python 3.11.
- Recreated `.venv` using `python3.11 -m venv .venv` and installed project requirements from the Week 9 solution `requirements.txt`.
- Updated VS Code workspace interpreter and the notebook `kernelspec` so notebooks use the `.venv` Python 3.11 kernel.

Why this change
----------------
- Multiple virtual environments (e.g. `venv`, `.venv_temp`) caused confusion and editor/kernel mismatches.
- Python 3.14 on the machine produced package build failures for pinned wheels (pandas/numpy). Python 3.11 has compatible prebuilt wheels.

Repro steps (commands)
----------------------
1. Create the stable `.venv` (use Python 3.11):

   ```bash
   python3.11 -m venv .venv
   .venv/bin/python -m pip install --upgrade pip setuptools wheel
   ```

2. Install requirements for the Week 9 solution into `.venv`:

   ```bash
   .venv/bin/python -m pip install -r capstone_graded_project/week9_mini_project/week9_house_price_prediction/solution/requirements.txt
   ```

3. Set VS Code to use the `.venv` interpreter (already applied in workspace settings):

   - File: .vscode/settings.json
   - Key: `python.defaultInterpreterPath` → `${workspaceFolder}/.venv/bin/python`

4. Update notebook kernel metadata (done): ensure the notebook uses the workspace `.venv` kernel.

   - Notebook updated: `capstone_graded_project/week9_mini_project/week9_house_price_prediction/solution/week9_house_price_analysis.ipynb`

Verification
------------
- From repo root, check Python and imports inside the venv:

  ```bash
  .venv/bin/python --version
  .venv/bin/python -c "import pandas,numpy,sklearn,matplotlib,seaborn,nbclient,nbformat; print('env-check-ok')"
  ```

- Open the notebook in VS Code and restart the kernel (select the workspace `.venv` kernel if prompted). Re-run the import cell; it should run without errors.

Notes & Recommendations
-----------------------
- Avoid creating multiple venv directories in the project root. Keep a single `.venv` for the workspace and remove any temporary envs (`venv`, `.venv_temp`).
- If you need to run heavy builds, prefer a compatible Python version (3.11 for these pinned packages) so pip can use prebuilt wheels.
- Optionally, pre-install heavy ML libraries (xgboost, lightgbm, catboost) into `.venv` to avoid in-notebook `!pip install` calls.

Optional: Run the notebook programmatically
-----------------------------------------
- To run the notebook end-to-end and capture runtime errors, use `nbclient` inside the venv:

  ```bash
  .venv/bin/python -m pip install nbclient
  .venv/bin/python - <<'PY'
  from nbclient import NotebookClient
  import nbformat
  nb = nbformat.read('capstone_graded_project/week9_mini_project/week9_house_price_prediction/solution/week9_house_price_analysis.ipynb', as_version=4)
  client = NotebookClient(nb, timeout=600)
  client.execute()
  PY
  ```

Contact
-------
If you want, I can run the notebook end-to-end now (will execute training cells). Reply `run-notebook` to proceed.
