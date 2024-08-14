
from create_dist_multi_processing import run_mnist_experiment
from pathlib import Path
from robustness_experiment_box.verification_module.auto_verify_module import AutoVerifyModule
from robustness_experiment_box.verification_module.property_generator.one2any_property_generator import One2AnyPropertyGenerator
from autoverify.verifier import AbCrown
from definitions import AB_CROWN_CONFIG, RESULTS_ROOT, MNIST_NETWORK_FOLDER

def main():
    # Create distribution using AB-Crown verifier
    experiment_name = "first_100_images_multiprocessing_abcrown"
    timeout=600
    property_generator = One2AnyPropertyGenerator()
    verifier_module = AutoVerifyModule(verifier=AbCrown(), property_generator=property_generator, timeout=timeout, config=AB_CROWN_CONFIG)
    experiment_repository_path = Path(RESULTS_ROOT, 'MNIST')
    experiment_repository_path.mkdir(parents=True, exist_ok=True)
    run_mnist_experiment(verifier_module, experiment_name, 
                         network_folder_path=MNIST_NETWORK_FOLDER,
                         experiment_repository_path=experiment_repository_path, 
                         num_samples=100)



if __name__ == '__main__':
    main()