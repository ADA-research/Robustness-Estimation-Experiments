
from create_dist_multi_processing import run_mnist_experiment
from pathlib import Path
from robustness_experiment_box.verification_module.auto_verify_module import AutoVerifyModule
from robustness_experiment_box.verification_module.property_generator.one2any_property_generator import One2AnyPropertyGenerator
from autoverify.verifier import AbCrown
from robustness_experiment_box.database.experiment_repository import ExperimentRepository
from definitions import AB_CROWN_CONFIG, RESULTS_ROOT, MNIST_NETWORK_FOLDER

def main():

    experiment_repository = ExperimentRepository(Path(RESULTS_ROOT, 'MNIST/abcrown/generated'), MNIST_NETWORK_FOLDER)

    experiment_repository.load_experiment("first_100_images_multiprocessing_abcrown_12-08-2024+21_12")

    result_df = experiment_repository.get_result_df()

    computed_images = result_df.image_id.values
    missing_images = [x for x in range(0,100) if x not in computed_images]

    # Create distribution using AB-Crown verifier
    experiment_name = "first_100_images_multiprocessing_abcrown_mnist_relu_4_1024_continue"
    timeout = 600
    property_generator = One2AnyPropertyGenerator()
    verifier_module = AutoVerifyModule(verifier=AbCrown(), property_generator=property_generator, timeout=timeout, config=AB_CROWN_CONFIG)
    experiment_repository_path = Path(RESULTS_ROOT, 'MNIST')
    experiment_repository_path.mkdir(parents=True, exist_ok=True)
    run_mnist_experiment(verifier_module, experiment_name, 
                         network_folder_path=MNIST_NETWORK_FOLDER,
                         experiment_repository_path=experiment_repository_path, 
                         samples=missing_images,
                         network_index=2)



if __name__ == '__main__':
    main()