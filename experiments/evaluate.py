"""Versioned evaluation entry point. See experiments/eval/README.md."""
from pathlib import Path
import sys
PROJECT_DIRECTORY=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(PROJECT_DIRECTORY))
from dotenv import load_dotenv
load_dotenv(PROJECT_DIRECTORY/'.env')
from experiments.eval.runner import main, run_evaluation
if __name__=='__main__':sys.exit(main())
