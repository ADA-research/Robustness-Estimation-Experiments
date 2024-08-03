import logging

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

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

def init(l):
    global lock
    lock = l

def create_data(data_point: DataPoint, experiment_repository: ExperimentRepository, epsilon_value_estimator: EpsilonValueEstimator, network: Network):
    verification_context = experiment_repository.create_verification_context(network, data_point)

    epsilon_value_result = epsilon_value_estimator.compute_epsilon_value(verification_context)

    with lock:
        experiment_repository.save_result(epsilon_value_result)

def create_distribution(experiment_repository: ExperimentRepository, dataset: ExperimentDataset, dataset_sampler: DatasetSampler, epsilon_value_estimator: EpsilonValueEstimator):

    lock = Lock()
    network_list = experiment_repository.get_network_list()
    failed_networks = []
    for network in network_list:
        try:
            sampled_data = dataset_sampler.sample(network, dataset)
        except Exception as e:
            logging.info(f"failed for network: {network}, with exception: {e}")
            failed_networks.append(network)
            continue
        with multiprocessing.Pool(processes=4, initializer=init, initargs=(lock,)) as pool:
            pool.map(partial(create_data, experiment_repository=experiment_repository, epsilon_value_estimator=epsilon_value_estimator, network=network), sampled_data)

    experiment_repository.save_plots()
    logging.info(f"Failed for networks: {failed_networks}")

def main():

    timeout = 50
    epsilon_list = np.arange(0, 0.4, 1/255)
    experiment_repository_path = Path(f'./generated')
    network_folder = Path("./networks/mnist_networks")
    torch_dataset = torchvision.datasets.MNIST(root="../data", train=False, download=False, transform=transforms.ToTensor())

    dataset = PytorchExperimentDataset(dataset=torch_dataset)
    dataset = dataset.get_subset([x for x in range(0,100)])

    experiment_repository = ExperimentRepository(base_path=experiment_repository_path, network_folder=network_folder)

    property_generator = One2AnyPropertyGenerator()
    
    # Create distribution using AB-Crown verifier
    experiment_name = "first_100_images_multiprocessing_abcrown"
    verifier = AutoVerifyModule(verifier=AbCrown(), property_generator=property_generator, timeout=timeout, config=Path("experiments/abcrown_verification/ffn_config.yaml"))
    epsilon_value_estimator = BinarySearchEpsilonValueEstimator(epsilon_value_list=epsilon_list.copy(), verifier=verifier)
    dataset_sampler = PredictionsBasedSampler(sample_correct_predictions=True)
    experiment_repository.initialize_new_experiment(experiment_name)
    experiment_repository.save_configuration(dict(
                                        experiment_name=experiment_name, experiment_repository_path=str(experiment_repository_path),
                                        network_folder=str(network_folder), dataset=str(dataset),
                                        timeout=timeout, epsilon_list=[str(x) for x in epsilon_list]))

    create_distribution(experiment_repository, dataset, dataset_sampler, epsilon_value_estimator)

    

if __name__ == "__main__":
    main()