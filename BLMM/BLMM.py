import os
import sys
import shutil
import yaml
import numpy as np
from dask.distributed import as_completed
from BLMM.lib.fileio import  cluster_detection
from BLMM.src.blmm_setup import setup
from BLMM.src.blmm_batch import batch
from BLMM.src.blmm_cleanup import cleanup
from BLMM.src.blmm_concat import concat
from BLMM.src.blmm_results  import results


def _main(argv=sys.argv[1:]):
    
    
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
    maxmem = eval(inputs['MAXMEM'])
    client, cluster = cluster_detection(inputs["clusterType"] , maxmem)

    # --------------------------------------------------------------------------------
    # Run Setup
    # --------------------------------------------------------------------------------
    
    # Ask for a node for setup
    cluster.scale(1)
    # Submit the setup job and retun the result
    future_s = client.submit(setup, inputs_yml, pure=False)
    nb, nvb = future_s.result()

    # Delete the future object (NOTE: This is important! If you don't delete this dask
    # tries to rerun it every time you call the result function again, e.g. after each
    # stage of the pipeline).
    del future_s
    
    # --------------------------------------------------------------------------------
    # Run Batch Jobs
    # --------------------------------------------------------------------------------
    
    # Ask for a node for setup
    # Make sure we don't use more nodes than necesary 
    if numNodes > nb+1 or numNodes > nvb+1:
        cluster.scale(min([nb, nvb]) )
    else:
        cluster.scale(numNodes)
    # Empty futures list
    futures = []
    # Submit jobs
    for i in np.arange(1,nb+1):

        # Run the jobNum^{th} job.
        future_b = client.submit(batch, i, inputs_yml, pure=False)

        # Append to list 
        futures.append(future_b)

    # wait for results 
        completed = as_completed(futures)

        # Delete the future objects (NOTE: see above comment in setup section).
        del i, futures, future_b, completed
    
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

        # Delete the future objects (NOTE: see above comment in setup section).
        del i, futures, future_r, completed

    # --------------------------------------------------------------------------------
    # Run Cleanup Job
    # --------------------------------------------------------------------------------
    future_cl = client.submit(cleanup, inputs_yml, pure=False)
    future_cl.result()


