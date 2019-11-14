# BLM-py
This repository contains the code for Big Linear Models for Neuroimaging cluster and local usage.

## Requirements
To use the BLM-py code, `fsl 5.0.10` or greater must be installed and `fslpython` must be configured correctly. Alternatively the following python packages must be installed:

```
numpy>=1.14.0
nibabel>=2.2.1
yaml
pandas
subprocess
```

(This code may work with older versions of numpy and nibabel but caution is advised as these versions have not been tested).

If running `BLM-py` on a cluster, `fsl_sub` must also be configured correctly.

## Usage
To run `BLM-py` first specify your design using `blm_config.yml` and then run the job either in serial or in parallel by following the below guidelines.

### Specifying your model
The regression model for BLM must be specified in `blm_config.yml`. Below is a complete list of possible inputs to this file.

#### Mandatory fields
The following fields are mandatory:

 - `Y_files`: Text file containing a list of response variable images in NIFTI format.
 - `X`: CSV file of the design matrix (no column header, no ID row).
 - `outdir`: Path to the output directory.
 - `contrasts`: Contrast vectors to be tested. They should be listed as `c1,c2,...` etc and each contrast should contain the fields:
   - `name`: A name for the contrast. i.e. `AwesomelyNamedContrast1`.
   - `vector`: A vector for the contrast. This contrast must be one dimensional for a T test and two dimensional for an F test. For example; `[1, 0, 0]` (T contrast) or `[[1, 0, 0],[0,1,0]]` (F contrast).
   
   At least one contrast must be given, see `Examples` for an example of how to specify contrasts.
 
#### Optional fields

The following fields are optional:

 - `MAXMEM`: This is the maximum amount of memory (in bits) that the BLM code is allowed to work with. How this should be set depends on your machine capabilities; the default value however matches the SPM default of 2^32 (note this must be in python notation i.e. `2**32`).
 - `data_mask_files`: A text file containing a list of masks to be applied to the `Y_files`. 
   - The number of masks must be equal to the number of `Y_files` as each mask is applied to the corresponding entry `Y_files`. E.g. The first mask listed for `data_mask_files` will be applied to the first nifti in `Y_files`, the second mask in `data_mask_files` will be applied to the second nifti in `Y_files` and so on. 
 - `Missingness`: This field allows the user to mask the image based on how many studies had recorded values for each voxel. This can be specified in 3 ways.
   - `MinPercent`: The percentage of studies present at a voxel necessary for that voxel to be included in the final analysis mask. For example, if this is set to `0.1` then any voxel with recorded values for at least 10% of studies will be kept in the analysis.
   - `MinN`: The number of studies present at a voxel necessary for that voxel to be included in the final analysis mask. For example, if this is set to `20` then any voxel with recorded values for at least 20 studies will be kept in the analysis.
 - `analysis_mask`: A mask to be applied during analysis.
 - `OutputCovB`: If set to `True` this will output between beta covariance maps. For studies with a large number of paramters this may not be desirable as, for example, 30 analysis paramters will create 30x30=900 between beta covariance maps. By default this is set to `True`.
 - `data_mask_thresh`: Any voxel with value below this threshold will be treated as missing data. (By default, no such thresholding  is done, i.e. `data_mask_thresh` is essentially -infinity). 
 - `minlog`: Any `-inf` values in the `-log10(p)` maps will be converted to the value of `minlog`. Currently, a default value of `-323.3062153431158` is used as this is the most negative value which was seen during testing before `-inf` was encountered (see [this thread](https://github.com/TomMaullin/BLM/issues/76) for more details).


 
#### Examples

Below are some example `blm_config.yml` files.

Example 1: A minimal configuration.

```
Y_files: /path/to/data/Y.txt
X: /path/to/data/X.csv
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

An analysis can either be run in parallel on a computing cluster or in serial (one block after another). The below sections describe both scenarios.

#### Running an analysis in parallel

To run an analysis in parallel, log into the cluster you wish to run it on and ensure that `fsl` and `fsl_sub` are in the environment. On the `rescomp` cluster this can be done like so:

```
module add fsl
module add fsl_sub
```

Ensure you are in the `BLM-py` directory and once you are happy with the analysis you have specified in `blm_config.yml`, run the following command:

```
bash ./blm_cluster.sh
```

After running this you will see text printed to the commandline telling you the analysis is being set up and the jobs are being submitted. For large analyses or small cluster this may take a minute or two as several jobs may be submitted to the cluster. Once you can access the command line again, you can use `qstat` to see the jobs which have been submitted. You will typically see jobs of the following form:

 - `setup`: This will be working out the number of batches/blocks the analysis needs to be split into.
 - `batch*`: There may be several jobs with names of this format. These are the "chunks" the analysis has been split into. These are run in parallel to one another and typically don't take very long.
 - `results`: This code is combining the output of each batch to obtain statistical analyses. This will run once all `batch*` jobs have been completed. Please note this code has been streamlined for large numbers of subjects but not large number of parameters; therefore this job may take some time for large numbers of parameters.
 
#### Running an analysis in serial

To run an analysis in serial, ensure you are in the `BLM-py` directory and once you are happy with the analysis you have specified in `blm_config.yml`, run the following command:

```
fslpython -c "import blm_serial; blm_serial.main()"
```

The commandline will then tell you how much progress is being made as it runs each block.

### Analysis Output

Below is a full list of NIFTI files output after a BLM analysis.

| Filename  | Description  |
|---|---|
| `blm_vox_mask` | This is the analysis mask. |
| `blm_vox_n` | This is a map of the number of subjects which contributed to each voxel in the final analysis. |
| `blm_vox_edf` | This is the spatially varying error degrees of freedom mask. |
| `blm_vox_beta`  | These are the beta estimates.  |
| `blm_vox_con`  | These are the contrasts multiplied by the estimate of beta (this is the same as `COPE` in FSL).  |
| `blm_vox_cov`  | These are the between-beta covariance estimates.  |
| `blm_vox_conSE` | These are the standard error of the contrasts multiplied by beta (only available for T contrasts). |
| `blm_vox_conR2` | These are the partial R^2 maps for the contrasts (only available for F contrasts). |
| `blm_vox_resms` | This is the residual mean squares map for the analysis. |
| `blm_vox_conT` | These are the T statistics for the contrasts (only available for T contrasts). |
| `blm_vox_conF` | These are the F statistics for the contrasts (only available for F contrasts). |
| `blm_vox_conTlp` | These are the maps of -log10 of the uncorrected P values for the contrasts (T contrast). |
| `blm_vox_conFlp` | These are the maps of -log10 of the uncorrected P values for the contrasts (F contrast). |

The maps are given the same ordering as the inputs. For example, in `blm_vox_con`, the `0`th volume corresponds to the `1`st contrast, the `1`st volume corresponds to the `2`nd contrast and so on. For covariances, the ordering is of a similar form with covariance between beta 1 and  beta 1 (variance of beta 1) being the `0`th volume, covariance between beta 1 and  beta 2 being the `1`st volume and so on. In addition, a copy of the design is saved in the output directory as `inputs.yml`. It is recommended that this be kept for data provenance purposes.

## Testing

Note: All the below tests require access to test data. To ask access, please ask @TomMaullin directly.

### In parallel, against ground truth

To generate test cases:

```
bash ./generate_test_cases.sh $outdir $datadir
```

(Where `$datadir` is a data directory containg all data needed for analyses `test_cfg01.yml`, `test_cfg02.yml`,... `test_cfg10.yml` and `$outdir` is the desired output directory)

To check the logs:

```
bash ./check_logs.sh
```

To verify the test cases against ground truth:

```
bash ./verify_test_cases.sh $GTDIR
```

(Where `$GTDIR` is a directory containing ground truth data from a previous run, i.e. inside `$GTDIR` are the folders `test_cfg01.yml`, `test_cfg02.yml`, ... ect).

### In parallel, against FSL

To generate test cases:

```
bash ./generate_test_cases_fsl.sh $outdir $datadir
```

(Where `$datadir` is a data directory containg all data needed for designs `fsltest_cfg01.yml`, `fsltest_cfg02.yml` and `fsltest_cfg03.yml` and `$outdir` is the desired output directory)

To check the logs:

```
bash ./check_logs.sh
```

To verify the test cases against ground truth:

```
bash ./verify_test_cases_against_fsl.sh
```

### In serial

To test in serial, first run the parallel testing suite, and then afterwards simply run the following 3 test cases from the main `BLM-py` folder:

```
fslpython -c "import blm_serial; blm_serial.main('./test/cfg/test_cfg01_copy.yml')
fslpython -c "import blm_serial; blm_serial.main('./test/cfg/test_cfg02_copy.yml')
fslpython -c "import blm_serial; blm_serial.main('./test/cfg/test_cfg03_copy.yml')
```
