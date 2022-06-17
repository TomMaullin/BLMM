import sys
import os
import yaml
from dask import config
from dask_jobqueue import SGECluster
from dask.distributed import Client, as_completed
from dask.distributed import performance_report
from BLMM.src import blmm_setup


def _main(argv=None):
    """
    Main function
    """
    if len(sys.argv)>1:
        if not os.path.split(sys.argv[1])[0]:
            inputs_yml = os.path.join(os.getcwd(),sys.argv[1])
        else:
            inputs_yml = sys.argv[1]
    else:
        inputs_yml = os.path.join(os.path.realpath(__file__),'blm_config.yml')
    
    with open(inputs_yml, 'r') as stream:
        inputs = yaml.load(stream,Loader=yaml.FullLoader)
    
    numNodes = inputs['numNodes']
    # Timeouts
    config.set(distributed__comm__timeouts__tcp='90s')
    config.set(distributed__comm__timeouts__connect='90s')
    config.set(scheduler='single-threaded')
    config.set({'distributed.scheduler.allowed-failures': 50}) 
    config.set(admin__tick__limit='3h')
    cluster = SGECluster() # assumees dask configuration file
    client = Client(cluster)
    cluster.scale(numNodes)
    future_0 = client.submit(blmm_setup, inputs_yml, retnb, pure=False)
    nb = future_0.result()

if __name__ == "__main__":
    _main(sys.argv[1:])


