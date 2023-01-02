# House-price-prediction

## Introduction

In this report, I will elaborate how to construct a model to predict a home’s current market
value (as of a given date in time). The training dataset contains 11588 observations and the testing dataset constains 4402.
The features in the training dataset are listed below.

- PropertyID: Unique ID for home
- TransDate: Date of current sale
- SaledollarCnt: Price of current sale
- BathroomCnt: Number of bathrooms in home
- BedroomCnt: Number of bedrooms in home
- Builtyear: Year home was constructed
- FinishedSquarefeet: Finished square footage of the home
- GaragesquareFeet: Size of protected garage space if any
- LotsizeSquarefeet: Lot size of property in square feet
- StoryCnt: Number of stories for the home
- latitude: Latitude of the home * 1, 000, 000
- longitude: Longitude of the home * 1, 000, 000
- Usecode : Type of home (all homes in both training and test are single-family homes)
- Zonecodecount y : The intensity of use or density the lot is legally allowed to be built-up to
- viewtypeid: Nominal variable indicating the type of view from the home (blank or NULL value
indicates no view)
- censusblockgroup: The FIPS code for the census block group this property is located in. You can
derive the census tract FIPS by truncating the rightmost digit.
- BGMedHomevalue: The median home value in the block group
- BGMedRent: The median rent value in the block group
- BGMedYearBuilt: The median year structures in the block group were built
- BGPctown : Percentage of homes that are owner-occupied in the block group
- BGPctVacant: Percentage of housing that is vacant in the block group
- BGMedIncome: Median income of households residing in the block group
- BGPctKids : Percentage of households with children under 18 years present at home
- BGMedAge: Median age of residents of the block group

The ‘SaledollarCnt’ is the target. I build a model to predict the ‘SaledollarCnt’ of a
house using other features. The data prepossessing procedures are contained in Section
2 and models are discussed in Section 3

## Preprocessing
I removed the features ‘PropertyID’ and ‘Usecode’ because ‘PropertyID’ is unique for each
observation and ‘Usecode’ is the same for observations in training and testing dataset.
The feature ‘TransDate’ records the date of the current sale. I converted it to be the
number of days from ‘01/01/2015’ and scale it to be [0, 1] by dividing by 366. The feature
‘Viewtypeid’ contains many NA values that indicates no view, so I treated all NA values as
another category for this feature. For the feature ‘censusblockgroup’, I derived the census
tract FIPS by truncating the rightmost digit and treated it as a categorical variable.
For features ‘BGMedHomevalue’ and ‘BGMedIncome’, they are quite right-skewed, so I
made a log transformation on these two variables. The target ‘SaledollarCnt’ is also quite
right-skewed, so I used the log(‘SaledollarCnt’) as the response in my modeling procedure.
Among all these features, features ‘censusblockgroup’, ‘Zonecodecount’ and ‘viewtypeid’
are considered as categorical variables, and the remaining features are considered as
numerical variables.
For categorical variables, I provided two different options to deal with them. In my
first option, I applied the one-hot encode method on each categorical variable and treat
the new category in testing data as ‘unknown’. Since the feature ‘censusblockgroup’
has 376 categories and the the feature ‘Zonecodecount’ has 178 categories, the one-hot
encoding may lead to too many dimension increase for data. Thus, in my second option,
I applied the Multiple correspondence analysis (MCA) to use low dimensional embedding
to represent the categorical features ‘censusblockgroup’ and ‘Zonecodecount’. I choose
the number of components in MAC to be 8 (This number can be fine tuned) in my code.
The feature ‘ViewType’ is encoded by one-hot encoding as it only has 8 categories.
For numerical features, there are missing data in variables ‘GarageSquareFeet’, ‘BGMedRent’
and ‘BGMedYearBuilt’. I imputed the missing data by using the KNNimputer in
Sklearn package. Then, I scale all numerical features by Standscaler method in Sklearn.
The prepossessing procedure of the data is written to be a pipeline in my code. It
will be fitted on the training data then transform the testing data.

## Models
I applied two different kinds of models to predict the log(‘SaledollarCnt’). They are the
gradient boosting method and the XGBoost. The Gradient boosting method is achieved
by the ‘Sklearn’ package in Python and the XGBoost method is achieved by the ‘xgboost’
package. The dataset is splited into the training set and the validation set by using
‘train test split’ function in Sklearn. These models are fitted on the training dataset, then
compared on the validation dataset. For each model, there are a couple of hypeparameters
that need to be tuned, e.g. subsample, colsample bytree, max depth, min child weight
and reg lambda for the XGboost method. These hypeparameters are selected by using
the 5-fold cross-validation that is achieved by the GridSearchCV function in Sklearn.
Since here I applied multiple different models, I also considered the Superleaner ensemble
approach to combine those models.

