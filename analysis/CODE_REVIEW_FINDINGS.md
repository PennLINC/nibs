# Code Review Findings: analysis/ Directory

Analysis of 11 Python files for code quality, scientific correctness, and documentation accuracy.

---

## 1. build_missingness_list.py

### Hardcoded paths
- **Line 31**: `in_dir = '/cbica/projects/nibs/dset'`

### Bugs / Scientific correctness
- None identified.

### Comment/docstring mismatches
- None identified.

### Code elegance
- None significant.

### Undefined variables / Unused imports
- None identified.

---

## 2. build_brain_mask.py

### Hardcoded paths
- **Line 36**: `bids_dir = '/cbica/projects/nibs/dset'`
- **Line 37**: `deriv_dir = '/cbica/projects/nibs/derivatives'`
- **Line 38**: `work_dir = '/cbica/projects/nibs/work/brain_mask'`

### Bugs / Scientific correctness
- **Line 79 – `target_img` undefined**: `target_img` is only set when `modality == 'qsirecon'` (line 62). For modalities with `transforms is None` (e.g. `smriprep`), the code uses `target_img` in `mask.resample_image_to_target(target_img, ...)`. If the qsirecon file is missing for that subject/session, the loop `continue`s before setting `target_img`, causing a `NameError` when processing smriprep or qsm.
- **Lines 66–70 – mutation of shared transform list**: `transforms = mod_transforms[modality]` gets a reference to the list in the dict. Assigning `transforms[0] = ...` mutates the original `mod_transforms`, so later subjects/sessions get incorrect paths.
- **Docstring vs. implementation**: Docstring says “limit it to the intersection of all the masks”, but the code sums masks (`sum_mask = sum_mask + mask`), which implements a union, not an intersection. For an intersection, masks should be combined with a logical AND (e.g. multiplication and thresholding).

### Comment/docstring mismatches
- Docstring (lines 1–5): “intersection of all the masks” does not match the union behavior of the code.

### Code elegance
- None significant.

### Undefined variables / Unused imports
- None identified.

---

## 3. calculate_brain_mask_dice.py

### Hardcoded paths
- **Line 80**: `bids_dir = '/cbica/projects/nibs/dset'`
- **Line 81**: `deriv_dir = '/cbica/projects/nibs/derivatives'`
- **Line 82**: `work_dir = '/cbica/projects/nibs/work/brain_mask'`

### Bugs / Scientific correctness
- **Lines 128–133 – mutation of shared transform list**: Same pattern as `build_brain_mask.py`; mutating `transforms` mutates `mod_transforms`.
- **Lines 158–160 – possible `NameError`**: `smriprep_sum_mask` and `sum_mask` are only set inside loops. If no smriprep files exist, `smriprep_sum_mask` is never set; if no scalar modality files exist, `sum_mask` is never set. Both can cause `NameError` when writing outputs.

### Comment/docstring mismatches
- **Lines 1–5**: Docstring says “Create study-wide brain mask” but the script computes Dice coefficients and writes per-subject masks. It should describe Dice calculation and per-subject outputs.

### Code elegance
- **Line 87**: `masks = []` is defined but never used.

### Undefined variables / Unused imports
- `masks` (line 87) is unused.

---

## 4. generate_correlation_matrices.py

### Hardcoded paths
- **Line 10**: `bids_dir = '/cbica/projects/nibs/dset'`
- **Line 11**: `work_dir = '/cbica/projects/nibs/work/correlation_matrices'`
- **Line 13**: `out_dir = '../data'` (relative path)
- **Line 22**: `open('patterns.json', 'r')` (relative path; assumes CWD is `analysis/`)

### Bugs / Scientific correctness
- None identified. Fisher z-transform and averaging are used appropriately.

### Comment/docstring mismatches
- None identified.

### Code elegance
- None significant.

### Undefined variables / Unused imports
- None identified.

---

## 5. parcellate_scalar_maps.py

### Hardcoded paths
- **Line 194**: `bids_dir = '/cbica/projects/nibs/dset'`
- **Line 195**: `deriv_dir = '/cbica/projects/nibs/derivatives'`
- **Line 196**: `temp_dir = '/cbica/projects/nibs/work/correlation_matrices'`
- **Lines 197–202**: `target_file` hardcoded to `sub-22449/ses-01` for template resolution.

### Bugs / Scientific correctness
- **Lines 29–41 – variable reuse and redundant loads**: `qsirecon_brain_mask` is used as path, then overwritten with a binary mask. The same file is loaded three times (lines 38, 39, 40).
- **Line 173**: `if resampled_file:` followed by `pass` and commented-out `os.remove`; dead code.

### Comment/docstring mismatches
- **Lines 1–4**: Docstring says “Plot correlation matrices between myelin measures” but the script parcellates scalar maps into tissue masks. It should describe parcellation, not plotting.

### Code elegance
- Redundant file loading for the QSIRecon brain mask.
- Dead code around `resampled_file` (lines 173–175).

### Undefined variables / Unused imports
- None identified.

---

## 6. plot_correlation_matrices.py

### Hardcoded paths
- **Line 13**: `open('patterns.json', 'r')` (relative path)
- **Line 30**: `os.path.join('../data', filename)` (relative path)
- **Line 52**: `os.path.join('../figures', ...)` (relative path)

### Bugs / Scientific correctness
- None identified.

### Comment/docstring mismatches
- None identified.

### Code elegance
- **Line 17**: `modalities = modalities[::-1]` – `modalities` is never used after reversal.

### Undefined variables / Unused imports
- `modalities` (line 17) is assigned but never used.

---

## 7. plot_correlation_matrices_clustered.py

### Hardcoded paths
- **Line 23**: `pd.read_table('scalar_groups.tsv', ...)` (relative path)
- **Line 42**: `os.path.join('../data', filename)` (relative path)
- **Lines 86, 114**: `os.path.join('../figures', ...)` (relative path)

### Bugs / Scientific correctness
- None identified.

### Comment/docstring mismatches
- None identified.

### Code elegance
- **Lines 79, 105**: `xlabels = ax.ax_heatmap.get_xticklabels()` – assigned but never used.
- **Lines 66–75, 104–111**: Duplicated clustermap setup and tick configuration; could be factored into a helper.

### Undefined variables / Unused imports
- `xlabels` (lines 79, 105) is unused.

---

## 8. plot_correlation_matrices_mukherjee.py

### Hardcoded paths
- **Line 13**: `pd.read_table('scalar_groups.tsv', ...)` (relative path)
- **Line 52**: `os.path.join('../data', filename)` (relative path)
- **Line 86**: `os.path.join('../figures', ...)` (relative path)

### Bugs / Scientific correctness
- None identified.

### Comment/docstring mismatches
- None identified.

### Code elegance
- **Lines 36–40**: `to_flip` is defined but never used.

### Undefined variables / Unused imports
- `to_flip` (lines 36–40) is unused.

---

## 9. plot_correlation_matrix_diffs.py

### Hardcoded paths
- **Line 13**: `open('patterns.json', 'r')` (relative path)
- **Lines 29–34**: `os.path.join('../data', ...)` (relative path)
- **Lines 57–59**: `os.path.join('../figures', ...)` (relative path)

### Bugs / Scientific correctness
- None identified.

### Comment/docstring mismatches
- None identified.

### Code elegance
- **Line 17**: `modalities = modalities[::-1]` – `modalities` is never used.

### Undefined variables / Unused imports
- `modalities` (line 17) is unused.

---

## 10. plot_myelin_scalar_maps.py

### Hardcoded paths
- **Line 15**: `in_dir = '/cbica/projects/nibs/derivatives'`
- **Line 16**: `out_dir = '../figures'` (relative path)

### Bugs / Scientific correctness
- None identified.

### Comment/docstring mismatches
- **Lines 74–75**: Comment says “Temporarily skip” but the logic is `if 'G-Ratio' not in title: continue`, so only G-Ratio maps are processed. The comment should say something like “Temporarily only process G-Ratio maps”.

### Code elegance
- None significant.

### Undefined variables / Unused imports
- None identified.

---

## 11. utils.py

### Hardcoded paths
- None identified.

### Bugs / Scientific correctness
- None identified.

### Comment/docstring mismatches
- None identified.

### Code elegance
- **Line 70**: Parameter `filter=None` shadows the built-in `filter`. Consider renaming (e.g. `filter_type`, `filter_by`).

### Undefined variables / Unused imports
- None identified.

---

## Summary Table

| File | Hardcoded Paths | Bugs | Docstring Mismatch | Dead/Unused Code |
|------|-----------------|------|--------------------|------------------|
| build_missingness_list.py | 1 | 0 | 0 | 0 |
| build_brain_mask.py | 3 | 3 | 1 | 0 |
| calculate_brain_mask_dice.py | 3 | 2 | 1 | 1 |
| generate_correlation_matrices.py | 4 | 0 | 0 | 0 |
| parcellate_scalar_maps.py | 5 | 2 | 1 | 2 |
| plot_correlation_matrices.py | 3 | 0 | 0 | 1 |
| plot_correlation_matrices_clustered.py | 3 | 0 | 0 | 2 |
| plot_correlation_matrices_mukherjee.py | 3 | 0 | 0 | 1 |
| plot_correlation_matrix_diffs.py | 3 | 0 | 0 | 1 |
| plot_myelin_scalar_maps.py | 2 | 0 | 1 | 0 |
| utils.py | 0 | 0 | 0 | 1 |

---

## Recommendations

1. **Paths**: Move hardcoded paths to a config (e.g. `paths.yaml`) or environment variables.
2. **build_brain_mask.py**: Fix `target_img` usage (e.g. use smriprep mask as reference when qsirecon is missing), avoid mutating `mod_transforms`, and align implementation with the “intersection” docstring.
3. **calculate_brain_mask_dice.py**: Copy transform lists before mutating, add guards for empty data, and update the docstring.
4. **parcellate_scalar_maps.py**: Update docstring, reduce redundant file loads, and remove dead code around `resampled_file`.
5. **Unused variables**: Remove or use `modalities`, `to_flip`, `xlabels`, and `masks`.
