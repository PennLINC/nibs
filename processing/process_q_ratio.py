"""Calculate Q-Ratio (R1/R2*) from precomputed derivatives.

1. Collect R1 and R1-B1c from pymp2rage derivatives.
2. Collect R2*-E12345 and R2*-E2345 from megre derivatives.
3. Calculate Q-Ratio variants from these derivatives.

Make sure to follow any processing steps from Shim et al 2022
https://www.frontiersin.org/journals/neuroanatomy/articles/10.3389/fnana.2022.950650/full

even though this is from separate MP2RAGE and MEGRE scans instead of MEMP2RAGE.
"""
