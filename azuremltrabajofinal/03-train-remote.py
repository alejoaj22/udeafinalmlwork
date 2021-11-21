# 04-run-pytorch.py
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig

if __name__ == "__main__":
    #Connect to Azure ML WorkSpace
    ws = Workspace.from_config(path='./.azureml',_file_name='config.json')

    #Experiment
    experiment = Experiment(workspace=ws, name='day1-experiment-train-cifar')
    config = ScriptRunConfig(source_directory='./src',
                             script='train-remote.py',
                             compute_target='cpu-cluster-c')

    # set up cifar environment
    env = Environment.from_conda_specification(
        name='cifar-env',
        file_path='./.azureml/cifar-aml-env.yml'
    )

    config.run_config.environment = env

    #Execute experiment
    run = experiment.submit(config)

    #Print url
    aml_url = run.get_portal_url()
    print(aml_url)