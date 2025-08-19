"""Calculate the scaling factors for the MTsat and ihMTR measures to achieve a mean g-ratio of
0.7 in the splenium.

The g-ratio formula is

g-ratio = sqrt(FVF / (FVF + (MVF * scaling_factor)))

where MVF is the MTsat or ihMTR value. FVF is held constant, so we need to solve for the scaling factor.

The equation is solved using the mean g-ratio in the splenium across subjects.
I need to solve for scaling_factor so that g = 0.7.
The first step is to calculate the splenium mask,
then calculate mean FVF and MVF in the splenium across subjects.

g = sqrt(FVF / (FVF + (MVF * scaling_factor)))

(g ** 2) = FVF / (FVF + (MVF * scaling_factor))

(g ** 2) * (FVF + (MVF * scaling_factor)) = FVF

((g ** 2) * FVF) + ((g ** 2) * MVF * scaling_factor) = FVF

((g ** 2) * MVF * scaling_factor) = FVF - (FVF * (g ** 2))

((g ** 2) * MVF * scaling_factor) = FVF * (1 - (g ** 2))

scaling_factor = (FVF * (1 - (g ** 2))) / ((g ** 2) * MVF)

# Now set g to 0.7
scaling_factor = (FVF * (1 - (0.7 ** 2))) / ((0.7 ** 2) * MVF)

scaling_factor = (FVF * (1 - 0.49)) / (0.49 * MVF)

scaling_factor = (FVF * 0.51) / (0.49 * MVF)

# We need to do this for each MTsat- and ihMTR-derived g-ratio combination, namely
# MTsat+ISOVF/ICVF, MTsat+AWF, ihMTR+ISOVF/ICVF, and ihMTR+AWF.
"""
