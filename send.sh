#! /bin/sh


version=`cat version.txt`
echo "previous version $version"
version=`expr $version + 1`
echo "current version $version"

git tag -am "submission-v0.$version" submission-v0.$version
git push aicrowd master
git push aicrowd submission-v0.$version

echo $version > version.txt

git add version.txt
git commit -m"bump version to v0.$version"

