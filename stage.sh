#! /bin/sh


python local-to-prod.py

git add experiments/impala-github.yaml
git commit -m"AUTO: local->staging yaml"

git push gh master


