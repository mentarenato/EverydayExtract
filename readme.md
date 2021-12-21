# Everyday Extraction

## Quick install
Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) and create environment from dependencies:
```
conda env create -f environment.yml
conda activate everyday
```

Download [desired shape predictor](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat) and place the file into the project root.

## Running the script
Everything is glued together in `main.py`. Assuming photos and data are saved in directory `data`, run the following:
```
data/Everyday2.sqlite data/photos --save_landmarks
```