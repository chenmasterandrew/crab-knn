# crab-knn
implementation of the KNN algorithm in Python applied to biometrics data of purple rock crabs (leptograpsus variegatus) to classify the sex of leptograpsus crabs

## Write-up:
https://docs.google.com/document/d/1uK0CxXXiHFax31C6Azo5iRrv-vJqMtH2nyr85FKXDKc/edit?usp=sharing

## Dataset
https://www.kaggle.com/inputblackboxoutput/crab-body-metrics

## Setup environment (do this when first cloning the project)
```shell
git clone git@github.com:chenmasterandrew/crab-knn.git
cd crab-knn
py -3 -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

## Testing instructions
```shell
.venv/Scripts/activate
```
* set a value for constant K in crabtest.py
* run crabtest.py
