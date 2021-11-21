# 07-model-registration-azure.py
from azureml.core import Workspace
from azureml.core import Model

if __name__ == "__main__":
    ws = Workspace.from_config(path='./.azureml',_file_name='config.json')

    model = Model.register(model_name='house-price-model',
                           tags={'version': 'model1'},
                           model_path='output/my_model.h5',
                           workspace = ws)
    print(model.name, model.id, model.version, sep='\t')

    model = Model.register(model_name='house-price-model-scaler',
                           tags={'version': 'model1-scaler'},
                           model_path='output/scaler.pkl',
                           workspace = ws)
    print(model.name, model.id, model.version, sep='\t')