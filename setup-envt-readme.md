# Environment Setup Guide

This file describes how to create and activate a local Python environment for this AI/ML repository, install dependencies, and run the notebook successfully.

## 1. Create the virtual environment
From the repository root:
```bash
python3 -m venv venv
```

## 2. Activate the virtual environment
**macOS / Linux**
```bash
source venv/bin/activate
```

**Windows (PowerShell)**
```powershell
venv\Scripts\Activate.ps1
```

**Windows (cmd.exe)**
```cmd
venv\Scripts\activate.bat
```

## 3. Install dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If you are running the notebook that uses `xgboost` or `lightgbm`, install the OpenMP runtime on macOS first:
```bash
brew install libomp
```

Then reinstall or install the packages inside the virtual environment:
```bash
pip install xgboost lightgbm catboost
```

## 4. Verify the environment
```bash
python -c "import pandas, numpy, matplotlib, seaborn, scipy, sklearn; print('OK')"
```

## 5. Register the notebook kernel
Register the virtual environment as a Jupyter kernel so both browser and VS Code can use it:
```bash
python -m ipykernel install --user --name iit_madras_venv --display-name "IIT_Madras venv"
```

## 6. Run the notebook
From the repository root, with the virtual environment activated:
```bash
jupyter notebook
```

Then open:
`capstone_graded_project/week9_mini_project/week9_house_price_prediction/solution/week9_house_price_analysis.ipynb`
and run the cells.

### 6.1 Use the right kernel in VS Code
In VS Code, make sure the notebook is using the `IIT_Madras venv` kernel:
- Click the kernel selector at the top-right of the notebook editor
- Choose `IIT_Madras venv`
- Restart the kernel if needed

If VS Code still uses the wrong interpreter, also set the workspace Python interpreter to the venv:
- Open the Command Palette (Cmd+Shift+P)
- Run `Python: Select Interpreter`
- Choose `./venv/bin/python`

### 6.2 Run the notebook from the terminal
```bash
python -m jupyter nbconvert --to notebook --execute capstone_graded_project/week9_mini_project/week9_house_price_prediction/solution/week9_house_price_analysis.ipynb --output /tmp/week9_house_price_analysis_executed.ipynb
```

## 7. Common troubleshooting
- If you see `ModuleNotFoundError` in VS Code, the notebook kernel is not using `venv`.
- Confirm the Jupyter kernel matches the installed environment.
- If you see a Homebrew-managed Python error, make sure you are using the local `venv` and not `/opt/homebrew/bin/python3` directly.
- If the notebook still cannot find the dataset, verify the CSV file exists in the repository path and that the notebook is opened from the repository root.

## 8. Useful commands
```bash
# Activate environment
source venv/bin/activate

# Register the Jupyter kernel
python -m ipykernel install --user --name iit_madras_venv --display-name "IIT_Madras venv"

# Install missing package
pip install <package-name>

# Deactivate environment
deactivate
```
