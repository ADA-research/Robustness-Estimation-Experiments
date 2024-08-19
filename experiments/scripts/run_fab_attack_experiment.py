from dataclasses import asdict, dataclass
import time
from autoattack.fab_pt import FABAttack_PT
import torch
from torch import nn

import logging

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

import pandas as pd
from pathlib import Path
import numpy as np
import torch
torch.manual_seed(0)
import torchvision
import torchvision.transforms as transforms
from autoverify.verifier import AbCrown
import multiprocessing
from functools import partial
from multiprocessing import Lock
from definitions import DATASETS_ROOT, MNIST_NETWORK_FOLDER, RESULTS_ROOT

from robustness_experiment_box.database.experiment_repository import ExperimentRepository
from robustness_experiment_box.dataset_sampler.dataset_sampler import DatasetSampler
from robustness_experiment_box.dataset_sampler.predictions_based_sampler import PredictionsBasedSampler
from robustness_experiment_box.epsilon_value_estimator.epsilon_value_estimator import EpsilonValueEstimator
from robustness_experiment_box.epsilon_value_estimator.binary_search_epsilon_value_estimator import BinarySearchEpsilonValueEstimator
from robustness_experiment_box.verification_module.auto_verify_module import AutoVerifyModule
from robustness_experiment_box.database.dataset.experiment_dataset import ExperimentDataset
from robustness_experiment_box.database.dataset.pytorch_experiment_dataset import PytorchExperimentDataset
from robustness_experiment_box.database.network import Network
from robustness_experiment_box.database.dataset.data_point import DataPoint
from robustness_experiment_box.verification_module.property_generator.one2any_property_generator import One2AnyPropertyGenerator


@dataclass
class FABResult:
     network_path: str
     image_id: int
     perturbation_size: float
     adversarial: bool
     duration: float


def save_result(result: dict, experiment_repo: ExperimentRepository, targeted) -> None:
        csv_name = 'fab_attack_results_df.csv'
        if targeted:
             csv_name = 'targeted_' + csv_name
        result_df_path = experiment_repo.get_results_path() / csv_name
        
        if result_df_path.exists():
            df = pd.read_csv(result_df_path, index_col=0)
            df.loc[len(df.index)] = asdict(result)
        else:
            df = pd.DataFrame([asdict(result)])
        df.to_csv(result_df_path)


def run_mnist_experiment(network_folder_path=MNIST_NETWORK_FOLDER,
                         experiment_repository_path=Path(f'./generated'), 
                         samples=[x for x in range(0,5)], device='cpu'):
    
    experiment_name = 'first_100_images_fab_attack'
    torch_dataset = torchvision.datasets.MNIST(root=DATASETS_ROOT, train=False, download=True, transform=transforms.ToTensor())

    dataset = PytorchExperimentDataset(dataset=torch_dataset)
    dataset = dataset.get_subset([x for x in samples])

    experiment_repository = ExperimentRepository(base_path=experiment_repository_path, network_folder=network_folder_path)
    dataset_sampler = PredictionsBasedSampler(sample_correct_predictions=True)
    experiment_repository.initialize_new_experiment(experiment_name)
    experiment_repository.save_configuration(dict(
                                        experiment_name=experiment_name, experiment_repository_path=str(experiment_repository_path),
                                        network_folder=str(network_folder_path), dataset=str(dataset)))

    for targeted in [False, True]:
        network_list = experiment_repository.get_network_list()
        for network in network_list:
            model = network.load_pytorch_model()
            # settings as used in autoattack
            # TODO: eps is needed for random init, how should we choose it?
            fab = FABAttack_PT(model, n_restarts=5, n_iter=100, eps=0.3, seed=42,
                    norm='Linf', verbose=True, device=device, targeted=targeted)
            sampled_data = dataset_sampler.sample(network, dataset)
            
            for datapoint in sampled_data:
                model.zero_grad()

                start = time.time()
                perturbed_data = fab.perturb(datapoint.data,torch.tensor([datapoint.label]))
                duration = time.time() - start

                perturbation = perturbed_data - datapoint.data
                perturbation_size = perturbation.flatten().abs().max().item()

                output = model(perturbed_data)
                _, final_pred = output.max(1, keepdim=True)
                adversarial = (final_pred != datapoint.label).item()

                save_result(FABResult(network_path=network.path,
                                      image_id=datapoint.id, 
                                      perturbation_size=perturbation_size, 
                                      adversarial=adversarial, duration=duration), 
                                      experiment_repo=experiment_repository, 
                                      targeted=targeted)

if __name__ == 'main':
     run_mnist_experiment(network_folder_path=MNIST_NETWORK_FOLDER,
                         experiment_repository_path=Path(RESULTS_ROOT, 'MNIST'), 
                         samples=[x for x in range(0,100)], device='cuda')