import os
import sys
import shutil
import yaml
import numpy as np
from dask import config
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, as_completed
from dask.distributed import performance_report


def _main(argv=None):
    
    
    # --------------------------------------------------------------------------------
    # Check inputs
    # --------------------------------------------------------------------------------
    # Inputs file is first argument
    if len(argv)<1:
        raise ValueError('Please provide an inputs YAML file.')
        
    # Get the inputs filepath
    else:
        inputs_yml = argv[0]
        
    # Load the inputs yaml file
    with open(inputs_yml, 'r') as stream:
        inputs = yaml.load(stream,Loader=yaml.FullLoader)
    
    # --------------------------------------------------------------------------------
    # Read Output directory, work out number of batches
    # --------------------------------------------------------------------------------
    OutDir = inputs['outdir']
    
    # Get number of nodes
    numNodes = inputs['numNodes']

    # Need to return number of batches
    retnb = True
    
    # Work out if we are batching the image by voxels as well
    if 'voxelBatching' in inputs:
        voxelBatching = inputs['voxelBatching']
    else:
        voxelBatching = 0
    
    # --------------------------------------------------------------------------------
    # Set up cluster
    # --------------------------------------------------------------------------------
    if 'clusterType' in inputs:

        # Check if we are using a HTCondor cluster
        if inputs['clusterType'].lower() == 'htcondor':

            # Load the HTCondor Cluster
            from dask_jobqueue import HTCondorCluster
            cluster = HTCondorCluster()

        # Check if we are using an LSF cluster
        elif inputs['clusterType'].lower() == 'lsf':

            # Load the LSF Cluster
            from dask_jobqueue import LSFCluster
            cluster = LSFCluster()

        # Check if we are using a Moab cluster
        elif inputs['clusterType'].lower() == 'moab':

            # Load the Moab Cluster
            from dask_jobqueue import MoabCluster
            cluster = MoabCluster()

        # Check if we are using a OAR cluster
        elif inputs['clusterType'].lower() == 'oar':

            # Load the OAR Cluster
            from dask_jobqueue import OARCluster
            cluster = OARCluster()

        # Check if we are using a PBS cluster
        elif inputs['clusterType'].lower() == 'pbs':

            # Load the PBS Cluster
            from dask_jobqueue import PBSCluster
            cluster = PBSCluster()

        # Check if we are using an SGE cluster
        elif inputs['clusterType'].lower() == 'sge':

            # Load the SGE Cluster
            from dask_jobqueue import SGECluster
            cluster = SGECluster()

        # Check if we are using a SLURM cluster
        elif inputs['clusterType'].lower() == 'slurm':

            # Load the SLURM Cluster
            from dask_jobqueue import SLURMCluster
            cluster = SLURMCluster()

        # Check if we are using a local cluster
        elif inputs['clusterType'].lower() == 'local':

            # Load the Local Cluster
            from dask.distributed import LocalCluster
            cluster = LocalCluster()

        # Raise a value error if none of the above
        else:
            raise ValueError('The cluster type, ' + inputs['clusterType'] + ', is not recognized.')

    else:
        # Raise a value error if the cluster type was not specified
        raise ValueError('Please specify "clusterType" in the inputs yaml.')
    
    # --------------------------------------------------------------------------------
    # Connect to client
    # --------------------------------------------------------------------------------
    
    # Connect to cluster
    client = Client(cluster)   

    # --------------------------------------------------------------------------------
    # Run Setup
    # --------------------------------------------------------------------------------
    
    # Ask for a node for setup
    cluster.scale(1)
    
    # Submit the setup job and retun the result
    future_s = client.submit(setup, inputs_yml, pure=False)
    nb = future_s.result()

    # Delete the future object (NOTE: This is important! If you don't delete this dask
    # tries to rerun it every time you call the result function again, e.g. after each
    # stage of the pipeline).
    del future_s
    
    # --------------------------------------------------------------------------------
    # Run Batch Jobs
    # --------------------------------------------------------------------------------
    
    # Ask for a node for setup
    cluster.scale(numNodes)

    # Empty futures list
    futures = []

    # Submit jobs
    for i in np.arange(1,nb+1):

        # Run the jobNum^{th} job.
        future_b = client.submit(batch, i, inputs_yml, pure=False)

        # Append to list 
        futures.append(future_b)

    # Completed jobs
    completed = as_completed(futures)

    # Wait for results
    for i in completed:
        i.result()

    # Delete the future objects (NOTE: see above comment in setup section).
    del i, completed, futures, future_b
    
    # --------------------------------------------------------------------------------
    # Run Concatenation Job
    # --------------------------------------------------------------------------------

    # Submit concatenation job and wait for result
    future_c = client.submit(concat, inputs_yml, pure=False)
    future_c.result()

    # Delete the future objects (NOTE: see above comment in setup section).
    del future_c
    
    # --------------------------------------------------------------------------------
    # Run Results Jobs
    # --------------------------------------------------------------------------------
    
    # If we aren't voxel batching run the entire analysis in one serial job.
    if not voxelBatching:
    
        future_r = client.submit(results, inputs_yml, -1, pure=False)
        future_r.result()

        # Delete the future object (NOTE: see above comment in setup section).
        del future_r
        
    # If we are voxel batching, split the analysis into chunks and run in parallel.
    else:

        # Work out the number of voxel batches (this should have been output by the 
        # setup job)
        with open(os.path.join(OutDir,'nvb.txt')) as f:
            nvb = int(f.readline())

        # Empty futures list
        futures = []

        # Submit job for each voxel batch
        for i in np.arange(1,nvb+1):

            # Run the jobNum^{th} job.
            future_r = client.submit(results, inputs_yml, i, pure=False)

            # Append to list 
            futures.append(future_r)

        # Completed jobs
        completed = as_completed(futures)

        # Wait for results
        for i in completed:
            i.result()

        # Delete the future objects (NOTE: see above comment in setup section).
        del i, completed, futures, future_r
    
    # --------------------------------------------------------------------------------
    # Run Cleanup Job
    # --------------------------------------------------------------------------------
    future_cl = client.submit(cleanup, inputs_yml, pure=False)
    future_cl.result()
    
if __name__ == "__main__":
    _main(sys.argv[1:])


