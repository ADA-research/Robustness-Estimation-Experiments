import logging
import argparse
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

from pathlib import Path
import numpy as np
import torch
torch.manual_seed(0)
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset

from ada_verona.database.dataset.experiment_dataset import ExperimentDataset
from ada_verona.database.dataset.pytorch_experiment_dataset import PytorchExperimentDataset
from ada_verona.database.experiment_repository import ExperimentRepository
from ada_verona.dataset_sampler.dataset_sampler import DatasetSampler
from ada_verona.dataset_sampler.predictions_based_sampler import PredictionsBasedSampler
from ada_verona.epsilon_value_estimator.binary_search_epsilon_value_estimator import BinarySearchEpsilonValueEstimator
from ada_verona.epsilon_value_estimator.epsilon_value_estimator import EpsilonValueEstimator
from ada_verona.verification_module.attack_estimation_module import AttackEstimationModule
from ada_verona.verification_module.attacks.auto_attack_wrapper import AutoAttackWrapper
from ada_verona.verification_module.attacks.fgsm_attack import FGSMAttack
from ada_verona.verification_module.attacks.pgd_attack import PGDAttack
from ada_verona.verification_module.property_generator.one2any_property_generator import One2AnyPropertyGenerator
from ada_verona.verification_module.property_generator.property_generator import PropertyGenerator

CONFIG = {'cifar_7_1024': '.CIFAR/config/cifar_7_1024.yaml',
            'conv_big':'.CIFAR/config/cifar_conv_big.yaml',
            'resnet_4b': '.CIFAR/config/resnet_4b.yaml', 
            'resnet_18': '.CIFAR/config/cifar_resnet18.yaml'}

def get_balanced_sample(train_bool =True):
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)


    data_transforms = transforms.Compose([
    transforms.Resize([32,32]),
    transforms.ToTensor(), 
 
    ])
    if train_bool:
        torch_dataset= torchvision.datasets.CIFAR10(root="./data", train = True, download=True, transform=data_transforms)
           
    # Extract the labels
        labels = torch.tensor([torch_dataset[i][1] for i in range(len(torch_dataset))])

        # Use StratifiedShuffleSplit to create balanced subsets
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=100, random_state=seed)  
        for train_idx, _ in splitter.split(np.zeros(len(labels)), labels):
            balanced_sample_idx = train_idx  


    # else: 
    #     torch_dataset= torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=data_transforms)
           
    #     # Extract the labels
    #     labels = torch.tensor([torch_dataset[i][1] for i in range(len(torch_dataset))])

    #     # Use StratifiedShuffleSplit to create balanced subsets
    #     splitter = StratifiedShuffleSplit(n_splits=1, test_size=100, random_state=seed)  
    #     for _,test_idx in splitter.split(np.zeros(len(labels)), labels):
    #         balanced_sample_idx = test_idx  
        


    # # Create a subset of the original dataset using the balanced indices
    # balanced_dataset = Subset(torch_dataset, balanced_sample_idx)

    # return balanced_dataset
    
    else: 
        torch_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=data_transforms
        )
        
        # Extract the labels
        labels = torch.tensor([torch_dataset[i][1] for i in range(len(torch_dataset))])

        # # Use StratifiedShuffleSplit to create balanced subsets
        # splitter = StratifiedShuffleSplit(n_splits=1, test_size=100, random_state=seed)
        
        # for _, test_idx in splitter.split(np.zeros(len(labels)), labels):
        #     balanced_sample_idx = test_idx

        # # Reverse: keep all indices EXCEPT those in balanced_sample_idx
        # all_indices = np.arange(len(torch_dataset))
        # remaining_indices = np.setdiff1d(all_indices, balanced_sample_idx)

    # Create the reversed subset
    # reversed_dataset = Subset(torch_dataset, remaining_indices)
    return torch_dataset


def find_config(network_identifier, config_dict):
    for key, path in config_dict.items():
        if key in str(network_identifier.path):
            return path
    return None


def create_distribution(experiment_repository: ExperimentRepository, dataset: ExperimentDataset, dataset_sampler: DatasetSampler, epsilon_value_estimator: EpsilonValueEstimator, property_generator: PropertyGenerator):
    network_list = experiment_repository.get_network_list()
    failed_networks = []
    for network in network_list:

        #Update config in AbCrown
        epsilon_value_estimator.verifier.config = find_config(network, CONFIG)
        try:
            sampled_data = dataset_sampler.sample(network, dataset)
        except:
            logging.info(f"failed for network: {network}")
            failed_networks.append(network)
            continue
        for data_point in sampled_data:
            verification_context = experiment_repository.create_verification_context(network, data_point, property_generator)

            epsilon_value_result = epsilon_value_estimator.compute_epsilon_value(verification_context)

            experiment_repository.save_result(epsilon_value_result)

    experiment_repository.save_plots()
    logging.info(f"Failed for networks: {failed_networks}")





def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack_method', type=str, help = 'Name of the method we are using for finding adversarial examples.')
    parser.add_argument('--indices', type=int, help ='indice where to start the search.')
    parser.add_argument('--batch_size', type=int, help ='indice where to start the search.')    
    parser.add_argument('--start', type=int, help ='indice where to start the search.')
    args = parser.parse_args()
    

    attack_method = args.attack_method
    indice =  args.indices
    batch_size = args.batch_size

    


    #load dataset
    torch_dataset = get_balanced_sample(train_bool=False) #Added check to see whether the split is train

    #set per-query timeout and the list of epsilons to verify
    timeout = 3600
    epsilon_list = np.arange(0.00,0.4,0.0039)
    
    #TODO: change to correct path
    experiment_repository_path = Path(f'/path/to/results_LION')
    
    #TODO: change network folder path as necessary
    network_folder = Path("./CIFAR/networks")


    #Create dataset and experiment repo 
    dataset = PytorchExperimentDataset(dataset=torch_dataset)
    dataset = dataset.get_subset(range((indice*batch_size), ((indice*batch_size)+batch_size)))
    
    experiment_repository = ExperimentRepository(base_path=experiment_repository_path, network_folder=network_folder)

    # Create sampler for correct predictions or incorrect predictions
    dataset_sampler = PredictionsBasedSampler(sample_correct_predictions=True)
 
    #create the experiment based on a name you choose
    experiment_name = f"LION_CIFAR_{attack_method}_{indice}"

    experiment_repository.initialize_new_experiment(experiment_name)

    #Define verifier, we have used abCrown and the one2any property generator (untargeted verification)
    if attack_method == "fgsm":
        verifier = AttackEstimationModule(attack=FGSMAttack(), top_k=1)
    elif attack_method == "pgd":
        verifier = AttackEstimationModule(attack=PGDAttack(number_iterations=40), top_k=1)
    elif attack_method == "autoattack":
        verifier = AttackEstimationModule(attack=AutoAttackWrapper(), top_k=1)

    property_generator=One2AnyPropertyGenerator(number_classes=10, data_lb=0, data_ub=1)

    #initiate the binary search 
    epsilon_value_estimator= BinarySearchEpsilonValueEstimator(epsilon_value_list= epsilon_list.copy(), verifier=verifier)
    create_distribution(experiment_repository, dataset, dataset_sampler, epsilon_value_estimator, property_generator)

if __name__ == "__main__":
    main()