# tutorial/01-create-workspace.py
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core import Workspace

interactive_auth = InteractiveLoginAuthentication(tenant_id="99e1e721-7184-498e-8aff-b2ad4e53c1c2")
ws = Workspace.get(name='mlw-udea-esp-ml', 
    subscription_id='80b93596-a236-4e78-a08e-0642caf0149b', 
    resource_group='rg-mludea-class', 
    location='eastus', 
)

ws.write_config(path='.azureml')