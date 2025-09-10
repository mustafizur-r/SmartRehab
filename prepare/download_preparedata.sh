rm -rf SnapMoGen
mkdir SnapMoGen
cd SnapMoGen

echo -e "Downloading All necessary file"
gdown --fuzzy https://drive.google.com/file/d/1cSLBXo18bk6-inusvmf0iZCgLae4PomO/view?usp=sharing

echo -e "Unzipping preparedata.zip"
unzip preparedata.zip

echo -e "Cleaning preparedata.zip"
rm preparedata.zip

cd ../

echo -e "Downloading done!"