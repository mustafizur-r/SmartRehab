rm -rf HumanML3D
mkdir HumanML3D
cd HumanML3D

echo -e "Downloading HumanML3D dataset"
gdown --fuzzy https://drive.google.com/file/d/1Sy7wLsWnhOKs8nMvj5G4NXHvryauRWT7/view?usp=sharing

echo -e "Unzipping HumanML3D.zip"
unzip HumanML3D.zip

echo -e "Cleaning HumanML3D.zip"
rm HumanML3D.zip

cd ../

echo -e "Downloading done!"