#!/usr/bin/env bash
# Fetch the locked Xenium technical-replicate noise-floor source: Janesick 2023 breast, GSE243280.
# Rep1 = GSM7780153, Rep2 = GSM7780154 (same FFPE block, serial 5um sections, same 313-gene panel).
# WARNING: these are large (multi-GB) Xenium output bundles. Run on a machine with disk + bandwidth.
set -euo pipefail
OUT="${1:-data}"
mkdir -p "$OUT/rep1" "$OUT/rep2"

echo "Xenium breast replicates are distributed as 10x output bundles:"
echo "  Rep1: Xenium_FFPE_Human_Breast_Cancer_Rep1  (10x Genomics datasets portal)"
echo "  Rep2: Xenium_FFPE_Human_Breast_Cancer_Rep2"
echo "  GEO mirror: GSE243280  (GSM7780153 Rep1, GSM7780154 Rep2)"
echo
echo "Download the *_outs.zip for each replicate from the 10x portal (login may be required),"
echo "or via GEO supplementary files, then unzip into:"
echo "  $OUT/rep1   and   $OUT/rep2"
echo
echo "Each bundle must contain the Xenium 'gene_panel.json' (or panel metadata) so check_panel.py can"
echo "confirm both replicates use the identical 313-gene panel before building the noise floor."
echo
echo "Paired Visium + H&E for the same block: GSE243280 (CytAssist Visium) + the post-stain H&E image."
