# Text Sentence Analyzer
# Project description
All documentation, code, and notes can be found here, as well as links to other resources that I found helpful in successfully completing the program.
This algorithm detects the sentiment of a given text as Good, Bad or Neutral. For the configuration you have to add the dataset in the files folder, plus the two scripts.

The project is built using:
1. Python(Flask)
2. scikit-learn
3. pandas
 
**Setup**

import the libraries
```
pip install scikit-learn
```
```
pip install pandas
```
**Configuration**

```
from sklearn.feature_extraction.text import TfidfVectorizer
```
```
from sklearn.linear_model import PassiveAggressiveClassifier
```
```
from sklearn.metrics import accuracy_score,confusion_matrix
```
```
from sklearn.model_selection import train_test_split
```
```
from sklearn.model_selection import cross_val_score
```

```
import pickle
```

```
import joblib
```

Models to use for classification, I chose to use regretion, but for a supervised training I think it is ideal, feel free to improve the project.

*Classification*

1 Identifying which category an object belongs to.
2 Applications: Spam detection, image recognition.
3 Algorithms: SVM, nearest neighbors, random forest, 

*Regression*

1 Predicting a continuous-valued attribute associated with an object.
2 Applications: Drug response, Stock prices.
3 Algorithms: SVR, nearest neighbors, random forest,

# Project Description

in the 
...
files folder
...,contains the csi documents for training the model, with the primary data, add more data to have better accuracy.



