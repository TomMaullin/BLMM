import sys
import os
from unittest import result
import yaml
from dask import config, delayed, compute
from dask_jobqueue import SGECluster
from dask.distributed import Client, as_completed
from dask.distributed import performance_report
from BLMM.src import blmm_setup, blmm_batch, blmm_concat, blmm_results, blmm_cleanup


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
    cluster = SGECluster(memory="20Gb") # assumees dask configuration file
    client = Client(cluster)
    cluster.scale(numNodes)
    # blmm_setup gives back the number of batches
    future_0 = client.submit(blmm_setup.main, inputs_yml, pure=False)
    nb,nvb = future_0.result()
    # let's distribute batch jobs
    batch_future = []
    for job in range(1,nb+1):
        # blmm_batch.main(job,inputs_yml)
        fut = delayed(blmm_batch.main)(job,inputs_yml)
        batch_future.append(fut)
    compute(batch_future)

    # submit concatenation job
    blmm_concat.main(inputs_yml)
    # _ = client.submit(blmm_concat.main,inputs_yml,pure=False)
    # submit resutls job
    if nvb == 0:
        client.submit(blmm_results.main,inputs_yml,pure=False)
    else:
        result_future = []
        for job in range(1,nvb+1):
            fut = delayed(blmm_results.main)(inputs_yml, job, nb-1)
            result_future.append(fut)
        compute(result_future)
    # cleanup the temp dirs
    _ = client.submit(blmm_cleanup.main,inputs_yml)

if __name__ == "__main__":
    _main(sys.argv[1:])


