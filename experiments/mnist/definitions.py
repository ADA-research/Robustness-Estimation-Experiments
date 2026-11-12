from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

RESULTS_ROOT = Path('/path/to/results')
MNIST_NETWORK_FOLDER = REPO_ROOT / 'networks' / 'mnist_networks'
DATASETS_ROOT = Path('/path/to/datasets')
AB_CROWN_CONFIG = Path(__file__).resolve().parent / 'abcrown_verification' / 'ffn_config.yaml'
