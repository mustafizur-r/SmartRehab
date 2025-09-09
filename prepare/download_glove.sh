echo -e "Downloading glove (in use by the evaluators, not by MoMask++ itself)"
gdown --fuzzy https://drive.google.com/file/d/15nigMlgyvqMIaWBEqKz0q081bBwtG2hc/view?usp=sharing
rm -rf glove

unzip glove.zip
echo -e "Cleaning\n"
rm glove.zip

echo -e "Downloading done!"