rm -rf SnapMoGen
mkdir SnapMoGen
cd SnapMoGen

echo -e "Downloading All necessary file"
gdown --fuzzy https://drive.google.com/file/d/1cSLBXo18bk6-inusvmf0iZCgLae4PomO/view?usp=sharing

echo -e "Unzipping data.zip"
unzip data.zip

echo -e "Cleaning data.zip"
rm data.zip

echo -e "Downloading renamed_bvhs from SnapMoGen dataset"
gdown --fuzzy https://drive.google.com/file/d/13_k4CIByo9MZW5IwUBQrKWc4CNkUHd-B/view?usp=sharing

echo -e "Unzipping renamed_bvhs.zip"
unzip renamed_bvhs.zip

echo -e "Cleaning renamed_bvhs.zip"
rm renamed_bvhs.zip


echo -e "Downloading renamed_feats from SnapMoGen dataset"
gdown --fuzzy https://drive.google.com/file/d/1DADEF894kq02vYaoR9ss9hOyCTRUA4Tk/view?usp=sharing

echo -e "Unzipping renamed_feats.zip"
unzip renamed_feats.zip

echo -e "Cleaning renamed_feats.zip"
rm renamed_feats.zip

cd ../

echo -e "Downloading done!"