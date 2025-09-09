cd checkpoint_dir

cd humanml3d 
echo -e "Downloading evaluation models for HumanML3D dataset"
gdown --fuzzy https://drive.google.com/file/d/1HaoxKQVbQctcogr10OWt1kn0MMAUdBRg/view?usp=drive_link
echo -e "Unzipping humanml3d_evaluator.zip"
unzip humanml3d_evaluator.zip

echo -e "Clearning humanml3d_evaluator.zip"
rm humanml3d_evaluator.zip

cd ../snapmogen/
echo -e "Downloading pretrained models for SnapMoGen dataset"
gdown --fuzzy https://drive.google.com/file/d/1AxI86o9TXckb8mdNy0eulXclSBXxkwuP/view?usp=drive_link

echo -e "Unzipping snapmogen_evaluator.zip"
unzip snapmogen_evaluator.zip

echo -e "Clearning snapmogen_evaluator.zip"
rm snapmogen_evaluator.zip

cd ../../

echo -e "Downloading done!"