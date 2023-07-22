# BLMM-py
This repository contains the code for Big Linear Mixed Models for Neuroimaging cluster and local usage.

## Requirements
To use the BLMM code, please clone this repository to your cluster. 

```
git clone https://github.com/TomMaullin/BLMM.git
```

Then pip install the requirements:

```
pip install -r requirements.txt
```

Finally, you must set up your `dask-jobqueue` configuration file, which is likely located at `~/.config/dask/jobqueue.yaml`. This will require you to provide some details about your HPC system. See [here](https://jobqueue.dask.org/en/latest/configuration-setup.html#managing-configuration-files) for further detail. For instance, if you are using rescomp your `jobqueue.yaml` file may look something like this:

```
jobqueue:
  slurm:
    name: dask-worker

    # Dask worker options
    cores: 1                 # Total number of cores per job
    memory: "100GB"                # Total amount of memory per job
    processes: 1                # Number of Python processes per job

    interface: ib0             # Network interface to use like eth0 or ib0
    death-timeout: 60           # Number of seconds to wait if a worker can not find a scheduler
    local-directory: "/path/of/your/choosing/"       # Location of fast local storage like /scratch or $TMPDIR
    log-directory: "/path/of/your/choosing/"
    silence_logs: True

    # SLURM resource manager options
    shebang: "#!/usr/bin/bash"
    queue: short
    project: null
    walltime: '01:00:00'
    job-cpu: null
    job-mem: null
    log-directory: null

    # Scheduler options
    scheduler-options: {'dashboard_address': ':46405'}
```


If running the `BLMM` tests on a cluster, `fsl_sub` must also be configured correctly.

## Usage
To run `BLMM-py` first specify your design using `blmm_config.yml` and then run your analysis by following the below guidelines.

### Specifying your model
The regression model for BLMM must be specified in `blmm_config.yml`. Below is a complete list of possible inputs to this file.

#### Mandatory fields
The following fields are mandatory:

 - `Y_files`: Text file containing a list of response variable images in NIFTI format.
 - `analysis_mask`: A mask to be applied during analysis.
 - `X`: CSV file of the design matrix (no column header, no ID row).
 - `Z`: Random factors in the design. They should be listed as `f1,f2,...` etc and each random factor should contain the fields:
   - `name`: Name of the random factor.
   - `factor`: CSV file containing a vector of indices representing which level of the factor each image belongs to. e.g. if the first factor is `subject` and the second image belonged to subject 5, the second entry in this file should be 5.
   - `design`: CSV file containing the design matrix for this random factor.
 - `outdir`: Path to the output directory.
 - `contrasts`: Contrast vectors to be tested. They should be listed as `c1,c2,...` etc and each contrast should contain the fields:
   - `name`: A name for the contrast. i.e. `AwesomelyNamedContrast1`.
   - `vector`: A vector for the contrast. This contrast must be one dimensional for a T test and two dimensional for an F test. For example; `[1, 0, 0]` (T contrast) or `[[1, 0, 0],[0,1,0]]` (F contrast).
   
   At least one contrast must be given, see `Examples` for an example of how to specify contrasts.
 
#### Optional fields

The following fields are optional:

 - `MAXMEM`: This is the maximum amount of memory (in bits) that the BLMM code is allowed to work with. How this should be set depends on your machine capabilities; the default value however matches the SPM default of 2^32 (note this must be in python notation i.e. `2**32`).
 - `data_mask_files`: A text file containing a list of masks to be applied to the `Y_files`. 
   - The number of masks must be equal to the number of `Y_files` as each mask is applied to the corresponding entry `Y_files`. E.g. The first mask listed for `data_mask_files` will be applied to the first nifti in `Y_files`, the second mask in `data_mask_files` will be applied to the second nifti in `Y_files` and so on. 
 - `Missingness`: This field allows the user to mask the image based on how many studies had recorded values for each voxel. This can be specified in 3 ways.
   - `MinPercent`: The percentage of studies present at a voxel necessary for that voxel to be included in the final analysis mask. For example, if this is set to `0.1` then any voxel with recorded values for at least 10% of studies will be kept in the analysis.
   - `MinN`: The number of studies present at a voxel necessary for that voxel to be included in the final analysis mask. For example, if this is set to `20` then any voxel with recorded values for at least 20 studies will be kept in the analysis.
 - `OutputCovB`: If set to `True` this will output between beta covariance maps. For studies with a large number of paramters this may not be desirable as, for example, 30 analysis paramters will create 30x30=900 between beta covariance maps. By default this is set to `True`.
 - `data_mask_thresh`: Any voxel with value below this threshold will be treated as missing data. (By default, no such thresholding  is done, i.e. `data_mask_thresh` is essentially -infinity). 
 - `minlog`: Any `-inf` values in the `-log10(p)` maps will be converted to the value of `minlog`. Currently, a default value of `-323.3062153431158` is used as this is the most negative value which was seen during testing before `-inf` was encountered (see [this thread](https://github.com/TomMaullin/BLMM/issues/76) for more details).
 - `method`: (Beta option). Which method to use for parameter estimation. Options are: `pSFS` (pseudo Simplified Fisher Scoring), `SFS` (Simplified Fisher Scoring), `pFS` (pseudo Fisher Scoring) and `FS` (Fisher Scoring). The (recommended) default is `pSFS`.
 - `tol`: Tolerance for convergence for the parameter estimation. Estimates will be output once the log-likelihood changes by less than `tol` from iteration to iteration. The default value is `1e-6`. 
 - `voxelBatching`: (Recommended for large designs). If set to `1`, the parameter estimation and inference steps of the analysis will be performed on seperate groups (batches) of voxels concurrently/in parallel. By default this is set to `0`. This setting is purely for computation speed purposes.
 - `maxnvb`: (Only used when `voxelBatching` is set to `1`). The maximum number of voxel batches/concurrent jobs allowed for estimation and inference. By default this is set to `60`. For large designs, this prevents the code from trying to submit thousands of jobs, should it decide this would be the quickest way to perform computation. This setting is purely for computation speed purposes.
 - `maxnit`: The maximum number of iterations each voxel is allowed for parameter estimation. By default this is set to `10000` iterations. If the iteration limit is reached a warning is thrown in the log files.
 - `resms`: If set to `1`, the `blmm_vox_resms` volume is output, if set to `0`, the `blmm_vox_resms` volume is not output.
 - `safeMode`: If set to `1`, voxels with more random effects than observations will be dropped from the analysis. By default this is set to `1`. It is not recommended to change this setting without good reason.

 
#### Examples

Below are some example `blmm_config.yml` files.

Example 1: A minimal configuration.

```
Y_files: /path/to/data/Y.txt
X: /path/to/data/X.csv
Z:
  - f1: 
      name: factorName
      factor: /path/to/data/Z1factorVector.csv
      design: /path/to/data/Z1DesignMatrix.csv
outdir: /path/to/output/directory/
contrasts:
  - c1:
      name: Tcontrast1
      vector: [1, 0, 1, 0, 1]
```

Example 2: A configuration with multiple optional fields.

```
MAXMEM: 2**32
Y_files: /path/to/data/Y.txt
data_mask_files: /path/to/data/M_.txt
data_mask_thresh: 0.1
X: /path/to/data/X.csv
Z:
  - f1: 
      name: factorName
      factor: /path/to/data/Z1factorVector.csv
      design: /path/to/data/Z1DesignMatrix.csv
  - f2: 
      name: factorName2
      factor: /path/to/data/Z2factorVector.csv
      design: /path/to/data/Z2DesignMatrix.csv
outdir: /path/to/output/directory/
contrasts:
  - c1:
      name: Tcontrast1
      vector: [1, 0, 0]
  - c2:
      name: Tcontrast2
      vector: [0, 1, 0]
  - c3:
      name: Tcontrast3
      vector: [0, 0, 1]
  - c4:
      name: Fcontrast1
      vector: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
Missingness:
  MinPercent: 0.10
  MinN: 15
analysis_mask: /path/to/data/MNI152_T1_2mm_brain_mask.nii.gz
```

### Running the Analysis


On your HPC system, ensure you are in the `BLMM` directory and once you are happy with the analysis you have specified in `blmm_config.yml`, run the following command:

```
python blmm_cluster.py
```

You can watch your analysis progress either by using `qstat` or `squeue` (depending on your system), or by using the interactive dask console. To do so, in a seperate terminal, tunnel into your HPC as follows:

```
ssh -L <local port>:localhost:<remote port> username@hpc_address
```

where the local port is the port you want to view on your local machine and the remote port is the dask dashboard adress (for instance, if you are on rescomp and you used the above `jobqueue.yaml`, `<remote port>` is `46405`). On your local machine, in a browser you can now go to `http://localhost:<local port>/` to watch the analysis run.

### Analysis Output

Below is a full list of NIFTI files output after a BLMM analysis.

| Filename  | Description  |
|---|---|
| `blmm_vox_mask` | This is the analysis mask. |
| `blmm_vox_n` | This is a map of the number of input images which contributed to each voxel in the final analysis. |
| `blmm_vox_edf` | This is the spatially varying niave degrees of freedom\*. |
| `blmm_vox_beta`  | These are the beta (fixed effects parameter) estimates.  |
| `blmm_vox_sigma2`  | These are the sigma2 (fixed effects variance) estimates.  |
| `blmm_vox_D`  | These are the D (random effects variance) estimates\*\*. |
| `blmm_vox_con`  | These are the contrasts multiplied by the estimate of beta (this is the same as `COPE` in FSL).  |
| `blmm_vox_cov`  | These are the between-beta covariance estimates.  |
| `blmm_vox_conSE` | These are the standard error of the contrasts multiplied by beta (only available for T contrasts). |
| `blmm_vox_conR2` | These are the partial R^2 maps for the contrasts (only available for F contrasts). |
| `blmm_vox_resms` | This is the residual mean squares map for the analysis\*\*\*. |
| `blmm_vox_conT` | These are the T statistics for the contrasts (only available for T contrasts). |
| `blmm_vox_conF` | These are the F statistics for the contrasts (only available for F contrasts). |
| `blmm_vox_conTlp` | These are the maps of -log10 of the uncorrected P values for the contrasts (T contrast). |
| `blmm_vox_conT_swedf` | These are the maps of Sattherthwaithe degrees of freedom estimates for the contrasts (T contrast). |
| `blmm_vox_conFlp` | These are the maps of -log10 of the uncorrected P values for the contrasts (F contrast). |
| `blmm_vox_conF_swedf` | These are the maps of Sattherthwaithe degrees of freedom estimates for the contrasts (F contrast). |

The maps are given the same ordering as the inputs. For example, in `blmm_vox_con`, the `0`th volume corresponds to the `1`st contrast, the `1`st volume corresponds to the `2`nd contrast and so on. For covariances, the ordering is of a similar form with covariance between beta 1 and  beta 1 (variance of beta 1) being the `0`th volume, covariance between beta 1 and  beta 2 being the `1`st volume and so on. In addition, a copy of the design is saved in the output directory as `inputs.yml`. It is recommended that this be kept for data provenance purposes.

\* These degrees of freedom are not used in inference and are only given as reference. The degrees of freedom used in inference are the Sattherthwaite approximations given in `blmm_vox_conT_swedf`  and `blmm_vox_conF_swedf` .
\*\* The `D` estimates are ordered as `vech(D1)`,...,`vech(Dr)` where `Dk` is the Random effects covariance matrix for the `k`th random factor, `r` is the total number of random factors in the design and `vech` represents ["half-vectorisation"](https://en.wikipedia.org/wiki/Vectorization_(mathematics)#Half-vectorization).
\*\*\* This is optional and may differ from the estimate of `sigma2`, which accounts for the random effects variance.

### Model Comparison

`BLMM-py` also offers model comparison for nested single-factor models via Likelihood Ratio Tests under a `50:50` chi^2 mixture distribtuion assumption (c.f. Linear Mixed Models for Longitudinal Data. 2000. Verbeke, G. & Molenberghs, G. Chapter 6 Section 3.). To compare the output of two single-factor models in `BLMM-py` (or the output of a single-factor model from `BLMM-py` with the output of a corresponding linear model run using [`BLM-py`](https://github.com/TomMaullin/BLM)) run the following command:

```
bash ./blmm_compare.sh /path/to/the/results/of/the/smaller/model/ /path/to/the/results/of/the/larger/model/ /path/to/output/directory/
```

Below is a full list of NIFTI files output after a BLMM likelihood ratio comparison test.

| Filename  | Description  |
|---|---|
| `blmm_vox_mask` | This is the analysis mask (this will be the intersection of the masks from each analysis). |
| `blmm_vox_Chi2.nii` | This is the map of the Likelihood Ratio Statistic. |
| `blmm_vox_Chi2lp.nii` | This is the map of -log10 of the uncorrected P values for the likelihood ratio test. |


## Developer Notes

### Testing

Currently, only unit tests are available for `BLMM`. These can be accessed by in the `tests/Unit` folder and must be run from the top of the directory.

### Notation

Throughout the code, the following notation is universal.

 - `Y`: The response vector at each voxel.
 - `X`: The fixed effects design matrix.
 - `Z`: The random effects design matrix.
 - `sigma2`: (An estimate of) The fixed effects variance.
 - `beta`: (An estimate of) The fixed effects parameter vector.
 - `D`: (An estimate of) The random effects covariance matrix (in full).
 - `Ddict`: A dictionary containing the unique blocks of `D`. For example, `Ddict[k]` is the block of `D` representing within-factor covariance for the kth random factor. 
 - `XtX, XtY, XtZ, YtX, YtY, YtZ, ZtX, ZtY, ZtZ`: These are the product matrices (i.e. X transposed multiplied by X, X transposed multiplied by Y, etc...).
 - `n`: Number of observations/input images.
 - `r`: Number of Random Factors in the design.
 - `q`: Total number of Random Effects (duplicates included), i.e. the second dimension of, Z, the random effects design matrix.
 - `qu`: Total number of unique Random effects (`vech(D_1),...vech(D_r)`).
 - `p`: Number of Fixed Effects parameters in the design.
 - `nraneffs`: A vector containing the number of random effects for each factor, e.g. `nraneffs=[2,1]` would mean the first factor has 2 random effects and the second factor has 1 random effect.
 - `nlevels`: A vector containing the number of levels for each factor, e.g. `nlevels=[3,4]` would mean the first factor has 3 levels and the second factor has 4 levels.
 - `inputs`: A dictionary containing all the inputs from the `blmm_config.yml`.
 - `e`: The residual vector (i.e. `e=Y-X @ beta`)
 - `V`: The matrix `I+ZDZ'` where `I` is the identity matrix.
 - `DinvIplusZtZD`: The matrix `D(I+Z'ZD)^(-1)`.

The following subscripts are also common throughout the code:

 - `_sv`: Spatially varying. `a_sv` means we have a value of `a` for every voxel we are considering, or rather, `a` "varies" across space.
 - `_i`: "Inner" voxels. This refers to the set of voxels which do not have missingness caused by mask variability in their designs. Typically, these make up the vast majority of the brain and tend not to lie near the edge of the brain, hence "inner".
 - `_r`: "Ring" voxels. This refers to the set of voxels which have missingness caused by mask variability in their designs. Special care must be taken with these voxels as `X` and `Z` are not the same across this set. Typically, these make up a small minority of the brain and tend to lie near the edge of the brain; they look like a "ring" around the edge of the brain.
 - `2D`: A function or file with this suffix will contain code designed to work analysis only on one voxel. As `X`,`Y` and `Z` are all 2 dimensional, all arrays considered for one voxel are 2D, hence the suffix.
 - `3D`: A function or file with this suffix will contain code designed to work analysis on multiple voxels. As `X`,`Y` and `Z` are all 3 dimensional (an extra dimension has been added for "voxel number"), all arrays considered are 3D, hence the suffix.

When the user has specified 1 random factor and 1 random effect only, the matrices `DinvIplusZtZD` and `ZtZ` become diagonal. As a result of this, in this setting, instead of saving these variable as matrices of dimension `(v,q,q)` (one `(q,q)` matrix for every voxel), we only record the diagonal elements of these matrices. As a result, in this setting `DinvIplusZtZD` and `ZtZ` have dimension `(v,q)` throughout the code. This results in significant performance gains.

### Structure of the repository

The repository contains 4 main folders, plus 3 files at the head of the repository. These are:

 - `README.md`: This file.
 - `blmm_config.yml`: The file the user must enter their design into.
 - `blmm_cluster.sh`: The shell scipt used to run blmm (see previous).
 - `blmm_compare.sh`: The shell scipt used to run blmm likelihood ratio tests (see previous).
 - `lib`: Helper functions:
   - `npMatrix2d.py`: Helper functions for 2d numpy array operations.
   - `npMatrix3d.py`: Helper functions for 3d numpy array operations.
   - `cvxMatrix2d.py`: Helper functions for 2d cvxopt matrix operations (used only by `PeLS`).
   - `PeLS.py`: Code for the PeLS method (only for benchmarking, currently unavailable in BLMM).
   - `fileio.py`: Miscellenous functions for handling files.
   - `est2d.py`: Parameter estimation methods for inference on one voxel.
   - `est3d.py`: Parameter estimation methods for inference on multiple voxels.
 - `src`: The main stages of the blmm pipeline:
   - `blmm_setup`: Formats inputs and works out the number of batches needed.
   - `blmm_batch`: Calculates the product matrices for individual batches of images.
   - `blmm_concat`: Sums the product matrices across batches, to obtain the product matrices for the overall model. 
   - `blmm_results`: Seperate voxels into "Inner" and "Ring" and then calls to `blmm_estimate` and `blmm_inference`.
   - `blmm_estimate`: Estimates the parameters beta, sigma^2 and D.
   - `blmm_inference`: Performs statistical inference on parameters and outputs results.
   - `blmm_cleanup`: Removes any leftover files from the analysis.
   - `blmm_compare`: Performs likelihood ratio tests comparing the results of multiple analyses.
 - `test`: Test functions:
   - `Functional`: (WIP) Adapted from sister project `BLM`. Dummy analyses to check the changes to the code haven't affected the output.
   - `Unit`: Unit tests for individual parts of the code:
     - `genTestDat.py`: Functions to generate test datasets and product matrices.
     - `npMatrix2d_tests.py`: Unit tests for all functions in `npMatrix2d.py`.
     - `npMatrix3d_tests.py`: Unit tests for all functions in `npMatrix3d.py`.
     - `cvxMatrix2d_tests.py`: Unit tests for all functions in `cvxMatrix2d.py`.
     - `est2d_tests.py`: A function for comparing results of all methods in `est2d.py`, as well as `PeLS.py`.
     - `est3d_tests.py`: Functions for comparing results of all methods in `est3d.py`.
 - `scipts`: Bash scripts which run each individual stage of the BLMM pipeline.
