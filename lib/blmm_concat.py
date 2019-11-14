import warnings as w
# This warning is caused by numpy updates and should
# be ignored for now.
w.simplefilter(action = 'ignore', category = FutureWarning)
import numpy as np
import subprocess
import warnings
import resource
import nibabel as nib
import sys
import os
import glob
import shutil
import yaml
import time
import warnings
import subprocess
from lib.blm_eval import blm_eval
np.set_printoptions(threshold=np.nan)
from scipy import stats
from lib.blm_load import blm_load

# Developer notes:
# --------------------------------------------------------------------------
# In the following code I have used the following subscripts to indicate:
#
# _r - This means this is an array of values corresponding to voxels which
#      are present in between k and n_s-1 studies (inclusive), where k is
#      decided by the user specified thresholds. These voxels will typically
#      be on the edge of the brain and look like a "ring" around the brain,
#      hence "_r" for ring.
# 
# _i - This means that this is an array of values corresponding to voxels 
#      which are present in all n_s studies. These will usually look like
#      a smaller mask place inside the whole study mask. Hence "_i" for 
#      inner.
#
# _sv - This means this variable is spatially varying (There is a reading
#       per voxel). 
#
# --------------------------------------------------------------------------
# Author: Tom Maullin (04/02/2019)

def main(*args):

    # ----------------------------------------------------------------------
    # Check inputs
    # ----------------------------------------------------------------------
    if len(args)==0 or (not args[0]):
        # Load in inputs
        with open(os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    '..',
                    'blm_config.yml'), 'r') as stream:
            inputs = yaml.load(stream)
    else:
        if type(args[0]) is str:
            # In this case inputs file is first argument
            with open(os.path.join(args[0]), 'r') as stream:
                inputs = yaml.load(stream)
        else:  
            # In this case inputs structure is first argument.
            inputs = args[0]

    # ----------------------------------------------------------------------
    # Read basic inputs
    # ----------------------------------------------------------------------
    OutDir = inputs['outdir']
    
    # Get number of parameters
    c1 = blm_eval(inputs['contrasts'][0]['c' + str(1)]['vector'])
    c1 = np.array(c1)
    n_p = c1.shape[0]
    del c1
    
    # Read in the nifti size and work out number of voxels.
    with open(inputs['Y_files']) as a:
        nifti_path = a.readline().replace('\n', '')
        nifti = blm_load(nifti_path)

    NIFTIsize = nifti.shape
    n_v = int(np.prod(NIFTIsize))

    # ----------------------------------------------------------------------
    # Load X'X, X'Y, Y'Y and n_s
    # ----------------------------------------------------------------------
    if (len(args)==0) or (type(args[0]) is str):
        # Read the matrices from the first batch. Note XtY is transposed as np
        # handles lots of rows much faster than lots of columns.
        sumXtX = np.load(os.path.join(OutDir,"tmp","XtX1.npy"))
        sumXtY = np.load(os.path.join(OutDir,"tmp","XtY1.npy")).transpose()
        sumYtY = np.load(os.path.join(OutDir,"tmp","YtY1.npy"))
        nmapb  = blm_load(os.path.join(OutDir,"tmp", "blm_vox_n_batch1.nii"))
        n_s_sv = nmapb.get_data()

        # Delete the files as they are no longer needed.
        os.remove(os.path.join(OutDir,"tmp","XtX1.npy"))
        os.remove(os.path.join(OutDir,"tmp","XtY1.npy"))
        os.remove(os.path.join(OutDir,"tmp","YtY1.npy"))
        os.remove(os.path.join(OutDir,"tmp","blm_vox_n_batch1.nii"))

        # Work out how many files we need.
        XtX_files = glob.glob(os.path.join(OutDir,"tmp","XtX*"))

        # Cycle through batches and add together results.
        for batchNo in range(2,(len(XtX_files)+2)):

            # Sum the batches.
            sumXtX = sumXtX + np.load(
                os.path.join(OutDir,"tmp","XtX" + str(batchNo) + ".npy"))

            sumXtY = sumXtY + np.load(
                os.path.join(OutDir,"tmp","XtY" + str(batchNo) + ".npy")).transpose()

            sumYtY = sumYtY + np.load(
                os.path.join(OutDir,"tmp","YtY" + str(batchNo) + ".npy"))
            
            # Obtain the full nmap.
            n_s_sv = n_s_sv + blm_load(os.path.join(OutDir,"tmp", 
                "blm_vox_n_batch" + str(batchNo) + ".nii")).get_data()
            
            # Delete the files as they are no longer needed.
            os.remove(os.path.join(OutDir, "tmp","XtX" + str(batchNo) + ".npy"))
            os.remove(os.path.join(OutDir, "tmp","XtY" + str(batchNo) + ".npy"))
            os.remove(os.path.join(OutDir, "tmp","YtY" + str(batchNo) + ".npy"))
            os.remove(os.path.join(OutDir, "tmp", "blm_vox_n_batch" + str(batchNo) + ".nii"))

    else:
        # Read in sums.
        sumXtX = args[1]
        sumXtY = args[2].transpose()
        sumYtY = args[3]
        n_s_sv = args[4]

    # Save nmap
    nmap = nib.Nifti1Image(n_s_sv,
                           nifti.affine,
                           header=nifti.header)
    nib.save(nmap, os.path.join(OutDir,'blm_vox_n.nii'))
    n_s_sv = n_s_sv.reshape(n_v, 1)
    del nmap

    # Dimension bug handling
    if np.ndim(sumXtX) == 0:
        sumXtX = np.array([[sumXtX]])
    elif np.ndim(sumXtX) == 1:
        sumXtX = np.array([sumXtX])

    if np.ndim(sumXtY) == 0:
        sumXtY = np.array([[sumXtY]])
    elif np.ndim(sumXtY) == 1:
        sumXtY = np.array([sumXtY])

    # Get ns.
    X = blm_load(inputs['X'])
    n_s = X.shape[0]

    # ----------------------------------------------------------------------
    # Create Mask
    # ----------------------------------------------------------------------

    Mask = np.ones([n_v, 1])

    # Check for user specified missingness thresholds.
    if 'Missingness' in inputs:

        # Apply user specified missingness thresholding.
        if ("MinPercent" in inputs["Missingness"]) or ("minpercent" in inputs["Missingness"]):

            # Read in relative threshold
            if "MinPercent" in inputs["Missingness"]:
                rmThresh = inputs["Missingness"]["MinPercent"]
            else:
                rmThresh = inputs["Missingness"]["minpercent"]

            # If it's a percentage it will be a string and must be converted.
            rmThresh = str(rmThresh)
            if "%" in rmThresh:
                rmThresh = float(rmThresh.replace("%", ""))/100
            else:
                rmThresh = float(rmThresh)

            # Check the Relative threshold is between 0 and 1.
            if (rmThresh < 0) or (rmThresh > 1):
                raise ValueError('Minumum percentage missingness threshold is out of range: ' +
                                 '0 < ' + str(rmThresh) + ' < 1 violation')

            # Mask based on threshold.
            Mask[n_s_sv<rmThresh*n_s]=0

        if ("MinN" in inputs["Missingness"]) or ("minn" in inputs["Missingness"]):

            # Read in relative threshold
            if "minn" in inputs["Missingness"]:
                amThresh = inputs["Missingness"]["minn"]
            else:
                amThresh = inputs["Missingness"]["MinN"]

            # If it's a percentage it will be a string and must be converted.
            if isinstance(amThresh, str):
                amThresh = float(amThresh)

            # Mask based on threshold.
            Mask[n_s_sv<amThresh]=0

    # We remove anything with 1 degree of freedom (or less) by default.
    # 1 degree of freedom seems to cause broadcasting errors on a very
    # small percentage of voxels.
    Mask[n_s_sv<=n_p+1]=0

    if 'analysis_mask' in inputs:

        addmask_path = inputs["analysis_mask"]
        
        # Read in the mask nifti.
        addmask = blm_load(addmask_path).get_data().reshape([n_v,1])
        
        Mask[addmask==0]=0

    # Reshape sumXtX to correct n_v by n_p by n_p
    sumXtX = sumXtX.reshape([n_v, n_p, n_p])

    # We also remove all voxels where the design has a column of just
    # zeros.
    for i in range(0,n_p):
        Mask[np.where(sumXtX[:,i,i]==0)]=0

    # Remove voxels with designs without full rank.
    M_inds = np.where(Mask==1)[0]
    Mask[M_inds[np.where(
        np.absolute(blm_det(sumXtX[M_inds,:,:])) < np.sqrt(sys.float_info.epsilon)
        )]]=0

    # Output final mask map
    maskmap = nib.Nifti1Image(Mask.reshape(
                                    NIFTIsize[0],
                                    NIFTIsize[1],
                                    NIFTIsize[2]
                                    ),
                              nifti.affine,
                              header=nifti.header)
    nib.save(maskmap, os.path.join(OutDir,'blm_vox_mask.nii'))
    del maskmap

    # Get indices of voxels in ring around brain where there are
    # missing studies.
    R_inds = np.where((Mask==1)*(n_s_sv<n_s))[0]

    # Get indices of the "inner" volume where all studies had information
    # present. I.e. the voxels (usually near the middle of the brain) where
    # every voxel has a reading for every study.
    I_inds = np.where((Mask==1)*(n_s_sv==n_s))[0]
    del Mask

    # Number of voxels in ring
    n_v_r = R_inds.shape[0]

    # Number of voxels in inner mask
    n_v_i = I_inds.shape[0]

    # Create dpf map
    df_r = n_s_sv[R_inds,:] - n_p
    df_r = df_r.reshape([n_v_r])
    df_i = n_s - n_p

    # Unmask df
    df = np.zeros([n_v])
    df[R_inds] = df_r
    df[I_inds] = df_i

    df = df.reshape(int(NIFTIsize[0]),
                    int(NIFTIsize[1]),
                    int(NIFTIsize[2]))

    # Save beta map.
    dfmap = nib.Nifti1Image(df,
                            nifti.affine,
                            header=nifti.header)
    nib.save(dfmap, os.path.join(OutDir,'blm_vox_edf.nii'))
    del df, dfmap

    # ----------------------------------------------------------------------
    # Calculate betahat = (X'X)^(-1)X'Y and output beta maps
    # ----------------------------------------------------------------------    

    # Reshaping
    sumXtY = sumXtY.transpose()
    sumXtY = sumXtY.reshape([n_v, n_p, 1])

    # If we have indices where only some studies are present, work out X'X and
    # X'Y for these studies.
    if n_v_r:

        # Calculate masked X'X for ring
        sumXtX_r = sumXtX[R_inds,:,:]

        # Calculate masked X'Y for ring
        sumXtY_r = sumXtY[R_inds,:]

        # Calculate masked Beta for ring
        beta_r = np.linalg.solve(sumXtX_r, sumXtY_r)

        # Unmask Beta
        beta = np.zeros([n_v, n_p])

        # Outer ring values
        beta[R_inds,:] = beta_r.reshape([n_v_r, n_p])


    # If we have indices where all studies are present, work out X'X and
    # X'Y for these studies.
    if n_v_i:
        
        # X'X must be 1 by np by np for broadcasting
        sumXtX_i = sumXtX[I_inds[0],:,:]
        sumXtX_i = sumXtX_i.reshape([1, n_p, n_p])

        sumXtY_i = sumXtY[I_inds,:]

        # Calculate beta
        beta_i = np.linalg.solve(sumXtX_i, sumXtY_i)
        beta[I_inds,:] = beta_i.reshape([n_v_i,n_p])

    beta = beta.reshape([n_v, n_p]).transpose()

    beta_out = np.zeros([int(NIFTIsize[0]),
                         int(NIFTIsize[1]),
                         int(NIFTIsize[2]),
                         beta.shape[0]])

    # Cycle through betas and output results.
    for k in range(0,beta.shape[0]):

        beta_out[:,:,:,k] = beta[k,:].reshape(int(NIFTIsize[0]),
                                              int(NIFTIsize[1]),
                                              int(NIFTIsize[2]))

    # Save beta map.
    betamap = nib.Nifti1Image(beta_out,
                              nifti.affine,
                              header=nifti.header)
    nib.save(betamap, os.path.join(OutDir,'blm_vox_beta.nii'))
    del beta_out, betamap

    del sumXtY, sumXtX
    if n_v_r:
        del sumXtY_r
    if n_v_i:
        del sumXtY_i

    if np.ndim(beta) == 0:
        beta = np.array([[beta]])
    elif np.ndim(beta) == 1:
        beta = np.array([beta])

    # ----------------------------------------------------------------------
    # Calculate residual sum of squares e'e = Y'Y - (Xb)'Xb
    # ---------------------------------------------------------------------- 

    if n_v_i:

        # Reshape beta along smallest axis for quicker
        # residual calculation
        beta_i_t = beta_i.transpose(0,2,1)

        # Calculate Beta transpose times XtX and delete the
        # now redundant matrices.
        betatXtX_i = np.matmul(beta_i_t, sumXtX_i)
        del beta_i_t

        # Multiply BetatXtX by Beta and delete the redundant
        # matrices.
        betatXtXbeta_i = np.matmul(betatXtX_i, beta_i)
        del betatXtX_i

        # Reshape betat XtX beta
        betatXtXbeta_i = np.reshape(betatXtXbeta_i, [n_v_i,1])

        # Residual sum of squares
        ete_i = sumYtY[I_inds] - betatXtXbeta_i
        del betatXtXbeta_i

    if n_v_r:

        # Reshape beta along smallest axis for quicker
        # residual calculation
        beta_r_t = beta_r.transpose(0,2,1)

        # Calculate Beta transpose times XtX and delete the
        # now redundant matrices.
        betatXtX_r = np.matmul(beta_r_t, sumXtX_r)
        del beta_r_t

        # Multiply BetatXtX by Beta and delete the redundant
        # matrices.
        betatXtXbeta_r = np.matmul(betatXtX_r, beta_r)
        del betatXtX_r

        # Reshape betat XtX beta
        betatXtXbeta_r = np.reshape(betatXtXbeta_r, [n_v_r,1])

        # Residual sum of squares
        ete_r = sumYtY[R_inds] - betatXtXbeta_r
        del betatXtXbeta_r

    del sumYtY

    # ----------------------------------------------------------------------
    # Calculate residual mean squares = e'e/(n_s - n_p)
    # ----------------------------------------------------------------------

    # Unmask resms
    resms = np.zeros([n_v,1])

    # Mask spatially varying n_s
    if n_v_r:
        n_s_sv_r = n_s_sv[R_inds,:]

        # In spatially varying the degrees of freedom
        # varies across voxels
        resms_r = ete_r/(n_s_sv_r-n_p)
        resms[R_inds,:] = resms_r

    if n_v_i:

        # All voxels in the inner mask have n_s scans present
        resms_i = ete_i/(n_s-n_p)
        resms[I_inds,:] = resms_i

    resms = resms.reshape(NIFTIsize[0], 
                          NIFTIsize[1],
                          NIFTIsize[2])

    # Output ResSS.
    msmap = nib.Nifti1Image(resms,
                            nifti.affine,
                            header=nifti.header)
    nib.save(msmap, os.path.join(OutDir,'blm_vox_resms.nii'))
    del msmap, resms

    # ----------------------------------------------------------------------
    # Calculate beta covariance maps
    # ----------------------------------------------------------------------
        
    # Calculate masked (x'X)^(-1) values for ring
    if n_v_r:
        isumXtX_r = blm_inverse(sumXtX_r, ouflow=True)
    if n_v_i:
        isumXtX_i = blm_inverse(sumXtX_i, ouflow=True)

    if "OutputCovB" in inputs:
        OutputCovB = inputs["OutputCovB"]
    else:
        OutputCovB = True

    if OutputCovB:
        
        vol = 0
        covbetaij_out = np.zeros([int(NIFTIsize[0]),
                                  int(NIFTIsize[1]),
                                  int(NIFTIsize[2]),
                                  n_p*n_p])

        # Output variance for each pair of betas
        for i in range(0,n_p):
            for j in range(0,n_p):

                    # Unmask cov beta ij
                    covbetaij = np.zeros([n_v])

                    if n_v_r: 
                        # Calculate masked cov beta ij for ring
                        covbetaij_r = np.multiply(
                            resms_r.reshape([resms_r.shape[0]]),
                            isumXtX_r[:,i,j])
                        covbetaij[R_inds] = covbetaij_r
        
                    if n_v_i:
                        # Calculate masked cov beta ij for inner
                        covbetaij_i = np.multiply(
                            resms_i.reshape([resms_i.shape[0]]),
                            isumXtX_i[:,i,j])
                        covbetaij[I_inds] = covbetaij_i

                    covbetaij_out[:,:,:,vol] = covbetaij.reshape(
                                            NIFTIsize[0],
                                            NIFTIsize[1],
                                            NIFTIsize[2],
                                            )
                    vol = vol+1;
                        
        # Output covariance map
        covbetaijmap = nib.Nifti1Image(covbetaij_out,
                                       nifti.affine,
                                       header=nifti.header)
        nib.save(covbetaijmap,
            os.path.join(OutDir, 
                'blm_vox_cov.nii'))
        del covbetaij, covbetaijmap, vol, covbetaij_out
        if n_v_r:
            del covbetaij_r
        if n_v_i:
            del covbetaij_i

    # ----------------------------------------------------------------------
    # Calculate COPEs, statistic maps and covariance maps.
    # ----------------------------------------------------------------------
    n_c = len(inputs['contrasts'])

    # Record how many T contrasts and F contrasts we have seen
    n_ct = 0
    n_cf = 0
    for i in range(0,n_c):

        # Read in contrast vector
        cvec = blm_eval(inputs['contrasts'][i]['c' + str(i+1)]['vector'])
        cvec = np.array(cvec)

        if cvec.ndim == 1:
            n_ct = n_ct + 1
        else:
            n_cf = n_cf + 1

    # Current number for contrast (T and F)
    current_n_ct = 0
    current_n_cf = 0

    # Setup 4d volumes to output
    cbeta = np.zeros([int(NIFTIsize[0]), int(NIFTIsize[1]), int(NIFTIsize[2]), n_c])
    se_t = np.zeros([int(NIFTIsize[0]), int(NIFTIsize[1]), int(NIFTIsize[2]), n_ct])
    stat_t = np.zeros([int(NIFTIsize[0]), int(NIFTIsize[1]), int(NIFTIsize[2]), n_ct])
    p_t = np.zeros([int(NIFTIsize[0]), int(NIFTIsize[1]), int(NIFTIsize[2]), n_ct])
    stat_f = np.zeros([int(NIFTIsize[0]), int(NIFTIsize[1]), int(NIFTIsize[2]), n_cf])
    p_f = np.zeros([int(NIFTIsize[0]), int(NIFTIsize[1]), int(NIFTIsize[2]), n_cf])
    r2_f = np.zeros([int(NIFTIsize[0]), int(NIFTIsize[1]), int(NIFTIsize[2]), n_cf])


    for i in range(0,n_c):

        # Read in contrast vector
        # Get number of parameters
        cvec = blm_eval(inputs['contrasts'][i]['c' + str(i+1)]['vector'])
        cvec = np.array(cvec)

        # Calculate C\hat{\beta}}
        if n_v_r:
            cbeta_r = np.matmul(cvec, beta_r)
        if n_v_i:
            cbeta_i = np.matmul(cvec, beta_i)
    
        if cvec.ndim == 1:
            statType='T'
            cvec = cvec.reshape([1,cvec.shape[0]])
        else:
            statType='F'

        if statType == 'T':

            # A T contrast has only one row so we can output cbeta here
            current_cbeta = np.zeros([n_v,1])
            if n_v_r:
                current_cbeta[R_inds,:] = cbeta_r
            if n_v_i:
                current_cbeta[I_inds,:] = cbeta_i

            cbeta[:,:,:,current_n_ct] = current_cbeta.reshape(
                                                    NIFTIsize[0],
                                                    NIFTIsize[1],
                                                    NIFTIsize[2]
                                                    )

            # Unmask to output
            covcbeta = np.zeros([n_v])

            if n_v_r:
                # Calculate c'(X'X)^(-1)c
                cvectiXtXcvec_r = np.matmul(
                    np.matmul(cvec, isumXtX_r),
                    np.transpose(cvec)).reshape(n_v_r)

                # Calculate masked cov(c\hat{\beta}) for ring
                covcbeta_r = cvectiXtXcvec_r*resms_r.reshape(n_v_r)
                covcbeta[R_inds] = covcbeta_r

            if n_v_i:
                # Calculate c'(X'X)^(-1)c
                cvectiXtXcvec_i = np.matmul(
                    np.matmul(cvec, isumXtX_i),
                    np.transpose(cvec))

                # Calculate masked cov(c\hat{\beta}) for inner
                covcbeta_i = cvectiXtXcvec_i*resms_i.reshape(n_v_i)
                covcbeta[I_inds] = covcbeta_i

            se_t[:,:,:,current_n_ct] = np.sqrt(covcbeta.reshape(
                                                    NIFTIsize[0],
                                                    NIFTIsize[1],
                                                    NIFTIsize[2]
                                                    ))

            del covcbeta

            # Unmask T stat
            tStatc = np.zeros([n_v])

            # Calculate masked T statistic image for ring
            if n_v_r:
                tStatc_r = cbeta_r.reshape(n_v_r)/np.sqrt(covcbeta_r)
                tStatc[R_inds] = tStatc_r

            if n_v_i:
                tStatc_i = cbeta_i.reshape(n_v_i)/np.sqrt(covcbeta_i)
                tStatc[I_inds] = tStatc_i

            stat_t[:,:,:,current_n_ct] = tStatc.reshape(
                                                    NIFTIsize[0],
                                                    NIFTIsize[1],
                                                    NIFTIsize[2]
                                                )


            # Unmask p for this contrast
            pc = np.zeros([n_v])

            # Work out p for this contrast
            if n_v_i:
                # Do this seperately for >0 and <0 to avoid underflow
                pc_i = np.zeros(np.shape(tStatc_i))
                pc_i[tStatc_i < 0] = -np.log10(1-stats.t.cdf(tStatc_i[tStatc_i < 0], df_i))
                pc_i[tStatc_i >= 0] = -np.log10(stats.t.cdf(-tStatc_i[tStatc_i >= 0], df_i))

                # Remove infs
                if "minlog" in inputs:
                    pc_i[np.logical_and(np.isinf(pc_i), pc_i<0)]=inputs['minlog']
                else:
                    pc_i[np.logical_and(np.isinf(pc_i), pc_i<0)]=-323.3062153431158

                pc[I_inds] = pc_i

            if n_v_r:
                # Do this seperately for >0 and <0 to avoid underflow
                pc_r = np.zeros(np.shape(tStatc_r))
                pc_r[tStatc_r < 0] = -np.log10(1-stats.t.cdf(tStatc_r[tStatc_r < 0], df_r[tStatc_r < 0]))
                pc_r[tStatc_r >= 0] = -np.log10(stats.t.cdf(-tStatc_r[tStatc_r >= 0], df_r[tStatc_r >= 0]))

                # Remove infs
                if "minlog" in inputs:
                    pc_r[np.logical_and(np.isinf(pc_r), pc_r<0)]=inputs['minlog']
                else:
                    pc_r[np.logical_and(np.isinf(pc_r), pc_r<0)]=-323.3062153431158

                pc[R_inds] = pc_r

            p_t[:,:,:,current_n_ct] = pc.reshape(
                                                NIFTIsize[0],
                                                NIFTIsize[1],
                                                NIFTIsize[2]
                                              )

            # Record that we have seen another T contrast
            current_n_ct = current_n_ct + 1


            del tStatc, pc
            if n_v_i:
                del tStatc_i, pc_i, covcbeta_i
            if n_v_r:
                del tStatc_r, pc_r, covcbeta_r


        if statType == 'F':

            # Get dimension of cvector
            q = cvec.shape[0]

            # Make (c'(X'X)^(-1)c)^(-1) unmasked
            icvectiXtXcvec = np.zeros([n_v, q*q])

            # Calculate c'(X'X)^(-1)c
            # (Note C is read in the other way around for F)
            if n_v_r:

                cvectiXtXcvec_r = np.matmul(
                    np.matmul(cvec, isumXtX_r),
                    np.transpose(cvec))

                # Cbeta needs to be nvox by 1 by npar for stacked
                # multiply.
                cbetat_r = cbeta_r.transpose(0,2,1)

                # Calculate masked (c'(X'X)^(-1)c)^(-1) values for ring
                icvectiXtXcvec_r = blm_inverse(cvectiXtXcvec_r, ouflow=True).reshape([n_v_r, q*q])
                icvectiXtXcvec[R_inds,:]=icvectiXtXcvec_r

            if n_v_i:

                cvectiXtXcvec_i = np.matmul(
                    np.matmul(cvec, isumXtX_i),
                    np.transpose(cvec))

                # Cbeta needs to be nvox by 1 by npar for stacked
                # multiply.
                cbetat_i = cbeta_i.transpose(0,2,1)

                # Calculate masked (c'(X'X)^(-1)c)^(-1) values for inner
                icvectiXtXcvec_i = blm_inverse(cvectiXtXcvec_i, ouflow=True).reshape([1, q*q])
                icvectiXtXcvec[I_inds,:]=icvectiXtXcvec_i

            icvectiXtXcvec = icvectiXtXcvec.reshape([n_v, q, q])

            # Save F statistic
            fStatc = np.zeros([n_v])

            # Calculate the numerator of the F statistic for the ring
            if n_v_r:
                Fnumerator_r = np.matmul(
                    cbetat_r,
                    np.linalg.solve(cvectiXtXcvec_r, cbeta_r))

                Fnumerator_r = Fnumerator_r.reshape(Fnumerator_r.shape[0])

                # Calculate the denominator of the F statistic for ring
                Fdenominator_r = q*resms_r.reshape([n_v_r])

                # Calculate F statistic.
                fStatc_r = Fnumerator_r/Fdenominator_r
                fStatc[R_inds]=fStatc_r

            # Calculate the numerator of the F statistic for the inner 
            if n_v_i:
                Fnumerator_i = np.matmul(
                    cbetat_i,
                    np.linalg.solve(cvectiXtXcvec_i, cbeta_i))

                Fnumerator_i = Fnumerator_i.reshape(Fnumerator_i.shape[0])

                # Calculate the denominator of the F statistic for inner
                Fdenominator_i = q*resms_i.reshape([n_v_i])

                # Calculate F statistic.
                fStatc_i = Fnumerator_i/Fdenominator_i
                fStatc[I_inds]=fStatc_i

            stat_f[:,:,:,current_n_cf] = fStatc.reshape(
                                               NIFTIsize[0],
                                               NIFTIsize[1],
                                               NIFTIsize[2]
                                           )

            del fStatc

            # Unmask p for this contrast
            pc = np.zeros([n_v])

            # Work out p for this contrast
            if n_v_i:
                pc_i = -np.log10(1-stats.f.cdf(fStatc_i, q, df_i))

                # Remove infs
                if "minlog" in inputs:
                    pc_i[np.logical_and(np.isinf(pc_i), pc_i<0)]=inputs['minlog']
                else:
                    pc_i[np.logical_and(np.isinf(pc_i), pc_i<0)]=-323.3062153431158

                pc[I_inds] = pc_i

            if n_v_r:
                pc_r = -np.log10(1-stats.f.cdf(fStatc_r, q, df_r))

                # Remove infs
                if "minlog" in inputs:
                    pc_r[np.logical_and(np.isinf(pc_r), pc_r<0)]=inputs['minlog']
                else:
                    pc_r[np.logical_and(np.isinf(pc_r), pc_r<0)]=-323.3062153431158

                pc[R_inds] = pc_r

            p_f[:,:,:,current_n_cf] = pc.reshape(
                                               NIFTIsize[0],
                                               NIFTIsize[1],
                                               NIFTIsize[2]
                                           )

            # Unmask partialR2.
            partialR2 = np.zeros([n_v])

            # Mask spatially varying n_s
            if n_v_r:
                n_s_sv_r = n_s_sv_r.reshape([n_v_r])

                # Calculate partial R2 masked for ring.
                partialR2_r = (q*fStatc_r)/(q*fStatc_r + n_s_sv_r - n_p)
                partialR2[R_inds] = partialR2_r

            if n_v_i:
                # Calculate partial R2 masked for inner mask.
                partialR2_i = (q*fStatc_i)/(q*fStatc_i + n_s - n_p)
                partialR2[I_inds] = partialR2_i

            r2_f[:,:,:,current_n_cf] = partialR2.reshape(
                                                       NIFTIsize[0],
                                                       NIFTIsize[1],
                                                       NIFTIsize[2]
                                                   )

            # Record that we have seen another F contrast
            current_n_cf = current_n_cf + 1

            del partialR2

    # Save contrast maps
    if n_ct:

        # Output standard error map
        secbetamap = nib.Nifti1Image(se_t,
                                      nifti.affine,
                                      header=nifti.header)
        nib.save(secbetamap,
            os.path.join(OutDir, 
                'blm_vox_conSE.nii'))
        del se_t, secbetamap

        # Output statistic map
        tStatcmap = nib.Nifti1Image(stat_t,
                                    nifti.affine,
                                    header=nifti.header)
        nib.save(tStatcmap,
            os.path.join(OutDir, 
                'blm_vox_conT.nii'))
        del stat_t, tStatcmap

        # Output pvalue map
        pcmap = nib.Nifti1Image(p_t,
                                nifti.affine,
                                header=nifti.header)
        nib.save(pcmap,
            os.path.join(OutDir, 
                'blm_vox_conTlp.nii'))  
        del pcmap, p_t

        # Output cbeta/cope map
        cbetamap = nib.Nifti1Image(cbeta,
                                   nifti.affine,
                                   header=nifti.header)
        nib.save(cbetamap,
            os.path.join(OutDir, 
                'blm_vox_con.nii'))
        del cbeta, cbetamap

    if n_cf:


        # Output statistic map
        fStatcmap = nib.Nifti1Image(stat_f,
                                    nifti.affine,
                                    header=nifti.header)
        nib.save(fStatcmap,
            os.path.join(OutDir, 
                'blm_vox_conF.nii'))
        del stat_f, fStatcmap

        # Output pvalue map
        pcmap = nib.Nifti1Image(p_f,
                                nifti.affine,
                                header=nifti.header)
        nib.save(pcmap,
            os.path.join(OutDir, 
                'blm_vox_conFlp.nii'))  
        del pcmap, p_f

        # Output statistic map
        partialR2map = nib.Nifti1Image(r2_f,
                                    nifti.affine,
                                    header=nifti.header)
        nib.save(partialR2map,
            os.path.join(OutDir, 
                'blm_vox_conR2.nii'))
        del partialR2map, r2_f

    # Clean up files
    if len(args)==0:
        os.remove(os.path.join(OutDir, 'nb.txt'))
    shutil.rmtree(os.path.join(OutDir, 'tmp'))

    w.resetwarnings()


# This function inverts matrix A. If ouflow is True,
# special handling is used to account for over/under
# flow. In this case, it assumes that A has non-zero
# diagonals.
def blm_inverse(A, ouflow=False):

    # Work out number of matrices and dimension of
    # matrices. I.e. if we have seven 3 by 3 matrices
    # to invert n_r = 7, d_r = 3.
    n_r = A.shape[0]
    d_r = A.shape[1]

    # If ouflow is true, we need to precondition A.
    if ouflow:

        # Make D to be filled with diagonal elements
        D = np.broadcast_to(np.eye(d_r), (n_r,d_r,d_r)).copy()

        # Obtain 1/sqrt(diagA)
        diagA = 1/np.sqrt(A.diagonal(0,1,2))
        diagA = diagA.reshape(n_r, d_r)

        # Make this back into diagonal matrices
        diaginds = np.diag_indices(d_r)
        D[:, diaginds[0], diaginds[1]] = diagA 

        # Precondition A.
        A = np.matmul(np.matmul(D, A), D)

    # np linalg inverse doesn't handle dim=[1,1]
    if np.ndim(A) == 1:
        iA = 1/A
    else:
        iA = np.linalg.solve(A, np.eye(d_r).reshape(1,d_r,d_r))

    if ouflow:

        # Undo preconditioning.
        iA = np.matmul(np.matmul(D, iA), D)

    return(iA)

# This function calculates the determinant of matrix A/
# stack of matrices A, with special handling accounting
# for over/under flow. 
def blm_det(A):


    # Precondition A.
    # Work out number of matrices and dimension of
    # matrices. I.e. if we have seven 3 by 3 matrices
    # to invert n_r = 7, d_r = 3.
    n_r = A.shape[0]
    d_r = A.shape[1]

    # Make D to be filled with diagonal elements
    D = np.broadcast_to(np.eye(d_r), (n_r,d_r,d_r)).copy()

    # Obtain 1/sqrt(diagA)
    diagA = 1/np.sqrt(A.diagonal(0,1,2))
    diagA = diagA.reshape(n_r, d_r)

    # Make this back into diagonal matrices
    diaginds = np.diag_indices(d_r)
    D[:, diaginds[0], diaginds[1]] = diagA 

    # Calculate DAD.
    DAD = np.matmul(np.matmul(D, A), D)

    # Calculate determinants.
    detDAD = np.linalg.det(DAD)
    detDD = np.prod(diagA, axis=1)
    
    # Calculate determinant of A
    detA = detDAD/detDD

    return(detA)

if __name__ == "__rain__":
    main()
