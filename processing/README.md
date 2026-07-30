# Processing Notes

I must run sMRIPrep, QSIPrep, and QSIRecon on CUBIC.
I am able to run MP2RAGE, ihMT, T1w/T2w, MESE, MEGRE, and g-ratio processing on CUBIC or my lab PC.
I must run QSM processing on my lab PC because CUBIC and PMACS lack the right MATLAB + toolbox setup.

I am able to connect to PMACS from CUBIC or my PC, but cannot connect my PC to CUBIC.

Current workflow:

1. Run sMRIPrep, QSIPrep, QSIRecon on CUBIC.
2. Transfer derivatives and raw data from CUBIC to PMACS.
3. Transfer derivatives and raw data from PMACS to PC.
4. Run MP2RAGE, ihMT, T1w/T2w, MESE, MEGRE, QSM, and g-ratio processing on PC.
5. Transfer derivatives from PC to PMACS.
6. Transfer derivatives from PMACS to CUBIC.
