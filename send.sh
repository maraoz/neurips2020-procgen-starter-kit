#! /bin/sh


version=`cat version.txt`
echo "previous version $version"
version=`expr $version + 1`
echo "current version $version"

echo "version name: $version-$1"

python local-to-prod.py

git add experiments/impala-baseline.yaml
git commit -m"AUTO: local->prod yaml for v0.$version-$1"

echo $version > version.txt

git add version.txt
git commit -m"AUTO: bump version to v0.$version-$1"

git tag -am "submission-v0.$version-$1" submission-v0.$version-$1
git push aicrowd master
git push gh master
git push aicrowd submission-v0.$version-$1


