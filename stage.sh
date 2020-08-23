#! /bin/sh


python local-to-prod.py

git add experiments/impala-github.yaml
git add experiments/impala-baseline.yaml
git commit -m"AUTO: local->staging yaml"

git push gh master


