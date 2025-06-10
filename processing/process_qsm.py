"""Process QSM data.

Steps:
1.  Average the magnitude images.
2.  Coregister the averaged magnitude to the preprocessed T1w image from sMRIPrep.
3.  Extract the average magnitude image brain by applying the sMRIPrep brain mask.
4.  Warp T1w mask from T1w space into the QSM space by applying the inverse of the coregistration
    transform.
5.  Apply the mask in QSM space to magnitude images.
6.  Run SEPIA by calling the MATLAB script.
7.  Warp SEPIA derivatives to MNI152NLin2009cAsym space.

Notes:

- This doesn't apply the X-separation method or use the T2* map.
"""
