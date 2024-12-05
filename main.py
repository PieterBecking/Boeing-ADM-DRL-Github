import subprocess

# Run Jupyter notebook with stdout/stderr redirected to suppress output
notebook_path = "notebooks/train_dqn_both.ipynb"
subprocess.run(["jupyter", "notebook", notebook_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
