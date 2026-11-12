
from create_dist_multi_processing import run_mnist_experiment
from pathlib import Path
from robustness_experiment_box.verification_module.auto_verify_module import AutoVerifyModule
from robustness_experiment_box.verification_module.property_generator.one2any_property_generator import One2AnyPropertyGenerator
from autoverify.verifier import AbCrown
from robustness_experiment_box.verification_module.attack_estimation_module import AttackEstimationModule
from robustness_experiment_box.verification_module.attacks.fgsm_attack import FGSMAttack
from definitions import AB_CROWN_CONFIG, RESULTS_ROOT, MNIST_NETWORK_FOLDER

def main():
    experiment_name = f"first_100_images_fgsm"
    attack_based_approximator = AttackEstimationModule(FGSMAttack())
    experiment_repository_path = Path(RESULTS_ROOT, 'MNIST')
    experiment_repository_path.mkdir(parents=True, exist_ok=True)

    run_mnist_experiment(attack_based_approximator, experiment_name, 
                        network_folder_path=MNIST_NETWORK_FOLDER,
                        experiment_repository_path=experiment_repository_path, 
                        samples=[x for x in range(0,100)], 
                        multiprocessing=False)



if __name__ == '__main__':
    main()