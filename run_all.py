import sys
import subprocess
from pathlib import Path

project_root = Path(__file__).resolve().parent
scripts = project_root / 'scripts'

def run_simple(step_name, script_name):

    print(f'\n --> Running: {step_name}')

    result = subprocess.run([sys.executable, str(scripts / script_name)])

    if result.returncode != 0:
        print(f'\n --------> Error in {script_name}. Aborting..')
        sys.exit(1)

def run_module(step_name, module_path):

    print(f'\n--> Running: {step_name}')

    result = subprocess.run([sys.executable, '-m', module_path])
    if result.returncode != 0:
        print(f' --------> Error in module {module_path}. Aborting.')
        sys.exit(1)


if __name__ == '__main__':

    print('\n --> Reproducing the full IR system...')

    # preprocessing ...

    run_simple('\n --> Preprocessing dataset', 'preprocessing.py')

    # indexes building ...

    run_simple('\n --> Building BM25 and TF-IDF indexes', 'indexes_building.py')

    # evaluation (requires relative imports)..
    
    run_module('\n --> Evaluating system performance', 'scripts.evaluating')

    print('\n --> All steps are completed successfully!')

