# Neural Decision Forest Enhancement For Session-based Recommendation
This is the official PyTorch implementation of *SR-PredictAO: Session-based Recommendation with High-Capability Predictor Add-On*.


## Reauirements
- Python 3.7
- Pytorch 1.13.1

## Structure of Code
- Module: Main part of the enhancement module
- Experiment Results: The experiment log for Data in *Performance Comparisons* part in the paper
- DIDN+NDF: Implementation for enriched DIDN with dataset, the parameter is set to be version of best HR@20 for Diginetica
- SGNNHN+NDF: Implementation of enriched SGNN-HN module with dataset, the parameter is set to be version of best HR@20 for Diginetica
- LESSR+NDF: Implementation of enriched LESSR module with dataset, the parameter is set for HR@20 and MRR@20 for Diginetica

## Usage
1. Install required envs
2. Download and extract the datasets
3. For changing parameter of the base module, just change the codes in main
4. For changing parameter of NDF enrichment, go to module assignment and change the parameters when assignment

## Use own base module or own dataset
1. Do the Encoder-Predictor split of the base module
2. Add NDF to the base module
3. Select ways of append results (var-signal selection or tuning para)
4. Prepare dataset as the paper sain
5. Run the enriched module