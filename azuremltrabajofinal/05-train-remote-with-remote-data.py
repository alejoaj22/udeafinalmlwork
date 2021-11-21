# 06-run-pytorch-data.py
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Dataset

if __name__ == "__main__":
    ws = Workspace.from_config(path='./.azureml',_file_name='config.json')
    datastore = ws.get_default_datastore()
    dataset = Dataset.File.from_files(path=(datastore, 'datasets/cifar10'))

    experiment = Experiment(workspace=ws, name='day2-experiment-train-cifar')

    config = ScriptRunConfig(
        source_directory='./src',
        script='train-remote.py',
        compute_target='cpu-cluster-c',
        arguments=[
            '--data_path', dataset.as_named_input('input').as_mount(),
            '--learning_rate', 0.003,
            '--momentum', 0.92],
    )
    # set up pytorch environment
    env = Environment.from_conda_specification(
        name='cifar-env',
        file_path='./.azureml/cifar-aml-env.yml'
    )
    config.run_config.environment = env

    run = experiment.submit(config)
    aml_url = run.get_portal_url()
    print("Submitted to compute cluster. Click link below")
    print("")
    print(aml_url)

    # Register model from run id
    run.wait_for_completion(show_output=True)
    model = run.register_model(model_name='cifar10',
                        tags={'area': 'vision'},
                        model_path='outputs/cifar10.pkl')
