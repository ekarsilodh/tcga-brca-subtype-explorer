# TCGA-BRCA four-class teaching subset

Prepared on 25 November 2026 from the instructor-supplied
`TCGA-BRCA_Dataset_new.csv`. The instructor identifies this as data used in:
Lodh et al., A stacked ensemble meta-classifier framework for PAM50 breast
cancer subtype prediction from gene expression signatures (2026).
https://doi.org/10.1007/s13721-026-00755-x

The paper reports TCGA-BRCA expression data obtained through GDC. This archive
uses the supplied CSV; it is not a fresh GDC download and the original acquisition
and preprocessing were not independently reconstructed.

## Observed original file
- 1,148 rows, 1,694 columns: 1,691 gene columns, sample, PAM50, Unnamed:.
- Unnamed: is the sequential exported row index 0–1147.
- 1,036 distinct participant IDs; 112 participants have two samples.
- 1,035 primary tumour samples (code 01), 113 solid-tissue normal samples (11).
- Labels: LumA 534, LumB 203, Basal 183, Normal 150, Her2 78.
- Normal comprises 37 code-01 samples and 113 code-11 samples. The origin and
  meaning of these labels cannot be established from the table alone.
- No missing values. Expression values range from 0 to 20.8112.
  This range does not establish the normalization or logarithm applied.

## Deterministic teaching preparation
1. Retain sample code 01, parsed from the fourth hyphen-separated ID component.
2. Retain recorded labels Basal, Her2, LumA and LumB.
3. Remove Unnamed:; preserve sample, gene values, labels and source row order.
4. Write a new CSV. Do not scale, log-transform, impute, rank genes, balance
   classes, or select predictors using outcomes during preparation.

Result: 998 rows x 1,693 columns, 998 unique participants, 1,691 numeric genes,
no missing/infinite values, no duplicated expression rows. Class counts are
LumA 534, LumB 203, Basal 183, Her2 78. All four labels are treated as recorded
classification targets. Only these four classes are in scope.

TCGA identifier interpretation:
https://docs.gdc.cancer.gov/Encyclopedia/pages/TCGA_Barcode/
https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/sample-type-codes

## Unresolved provenance and interpretation
The supplied file does not document the expression transformation, the method
used to choose these 1,691 genes, prior imputation, or the label-generation steps.
Do not infer these from the filename, value range or absence of missing values.
Do not apply another log transformation by default. Do not call these raw counts
or claim they are the 50 PAM50 genes. Confirm the preparation history with the
instructor when possible. If selection/preprocessing used all TCGA samples,
a later split cannot undo that earlier information use.

Proceeding is authorized for a classroom coding demonstration with these limits
explicitly documented. Do not present results as independent clinical validation,
new biomarker discovery, or a reproduction of the paper. PAM50 labels derived
from expression represent a label-prediction task, not an independent diagnosis.

## Integrity
Original CSV SHA-256: `ad7f49b5faf3cf5535fe3957c4ddf3c1de47e22b2bba23c465e517b494f84f8d`
Teaching CSV SHA-256: `dba3b5348ced3328bf31ee2be58a3399fd15d02a72648a0159e4f57876162388`

The original file, sample-level inclusion manifest and reproducible preparation
script are supplied outside the coding workspace in instructor_reference/.
