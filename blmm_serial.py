import numpy as np
import yaml
import os
from lib import blm_setup
from lib import blm_batch
from lib import blm_concat

def main(*args):

    print('Setting up analysis...')

    if len(args)==0:
        ipath = os.path.join(os.getcwd(),'blm_config.yml')
    else:
        ipath = os.path.abspath(args[0])
    
    # Load in inputs
    print(ipath)
    print(os.getcwd())
    print(os.path.abspath(args[0]))
    print(len(args))
    print(args[0])
    with open(ipath, 'r') as stream:
        inputs = yaml.load(stream)

    # Run the setup job to obtain the number of batches needed.
    # The second argument tells setup to return nB rather than save it
    nB = blm_setup.main(ipath, True)

    # Reload inputs (setup may have changed paths in inputs if they
    # were relative).
    with open(ipath, 'r') as stream:
        inputs = yaml.load(stream)

    # Run batch jobs and concatenate results
    print('Running batch 1/' + str(nB))
    sumXtX, sumXtY, sumYtY, sumnmap = blm_batch.main(1, inputs)

    for i in range(1, nB):
        print('Running batch ' + str(i+1) + '/' + str(nB))
        XtX, XtY, YtY, nmap = blm_batch.main(i+1, inputs)
        sumXtX = sumXtX + XtX
        sumXtY = sumXtY + XtY
        sumYtY = sumYtY + YtY
        sumnmap = sumnmap + nmap

    # Run concatenation job
    print('Combining batch results...')
    blm_concat.main(inputs, sumXtX, sumXtY, sumYtY, sumnmap)

    # Retrieve Output directory
    OutDir = inputs['outdir']

    # Print final message
    print('Distributed analysis complete. Please see "' + OutDir + '" for output.')

if __name__ == "__main__":
    main()
