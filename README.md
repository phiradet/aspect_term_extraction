# Aspect Term Detector
This repository contains Jupyter notebooks, demonstrating methods to detect aspect terms from reviews. Two types of classifiers are explored -- Conditional Random Fields (CRFs) and Random Forest.

### Content
1) [Data observation.ipynb](http://https://github.com/phiradet/aspect_term_extraction/blob/master/Data%20observation.ipynb "Data observation.ipynb") explores the training dataset in the `./data` directory.
2) [Prediction.ipynb](https://github.com/phiradet/aspect_term_extraction/blob/master/Prediction.ipynb "Prediction.ipynb") contains the pipeline of aspect term detector (with CRFs and Random Forest) and theirs precision, recall, and F1. 


## Getting Started
### Installing dependencies
```
$ export ENV_NAME="_some_name_"
$ conda create --name $ENV_NAME --file requirements.txt
$ source activate $ENV_NAME
$ jupyter notebook
$ python -m spacy download en_core_web_lg
```

