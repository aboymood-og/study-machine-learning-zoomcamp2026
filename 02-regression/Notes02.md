# Notes 02 - Regression

##### 01-car-price-intro.md
the scenario: a user want to sell a car and want to set the better and fair price to their car. The user describe the car and our model give the best price.
we will use this dataset; https://www.kaggle.com/datasets/CooperUnion/cardataset. have a lot of features, the most interesant are the target MSRP(price of a car in easy words). Plan->use all the other 15 features to predict this price.
We will do the project in several steps:
1. Get the data and do exploratory data analysis: just look at the data and try to learn more about it.
2. Prepare the dataset and train a linear regression model on it to predict the price of a car.
3. Go into the details of how linear regression is implemented — we will actually implement it ourselves.
4. Evaluate the quality of the model with RMSE. RMSE stands for root mean squared error, a metric for evaluating the quality of model predictions.
5. Do some feature engineering: the process of creating new features, new characteristics that we can use for our model.
6. Deal with numerical stability problems and see how to solve them with regularization.
7. Finally, use the model.

---

##### 02-data-preparation.md

here the things are more practical than any other stuff so i take note just for the important and new things. not for every shit.
we cannot aply string methods to numbers.
first normalize the df names columns and strings data.
strings are saved, normally, as object in pandas. but for the notebook that im using, are str XD?
Pandas attributes and methods (i copypaste them from notes for fast learning reasons.):
    pd.read_csv(<file_path_string>) -> read csv files
    df.head() -> take a look of the dataframe
    df.columns -> retrieve colum names of a dataframe
    df.columns.str.lower() -> lowercase all the letters
    df.columns.str.replace(' ', '_') -> replace the space separator
    df.dtypes -> retrieve data types of all features
    df.index -> retrieve indices of a dataframe

---

##### 3-eda.md

on eda we look each column and see what kind of values are there, watch the data, create hypothesis and problem before training a model.
.unique() returnr unique values in a column ([:5] until 5 but should be more if you want); .nunique() returns how many unique values are there.
the interasnt here runing a;
for col in df.columns:
    print(col)
    print(df[col].unique()[:5])
    print(df[col].nunique())
    print();
its understand the data and detect possible werid data like 3 doors cars, and so on. Understang what does each variable and the name of those.
then we check distribution of some variables, here for price (wich is what we are insterested for), to that we use graphs (import matplotlib.pyplot as plt and import seaborn as sns).
> %matplotlib inline is an IPython magic command used in Jupyter Notebooks to display Matplotlib plots directly below the code cell that generated them, rather than opening them in a separate window.
distribution of cars = how many cars cost what?. we use a histogram to do that. (sns.histplot(df.col, bins=50).
the 1e6 are cientific notation, means that num multiply by million in this case. again, know how read and understnd the histogram are the key of see this, on this part of the course for ex is the key to understd as mayority of cars are less of 50000 and probably 1 cost 500.000. 
long-tail distribution are that most of the data sits in one place and a shallow tail stretches far to the right. those are common on prices bc the gap between cheap and rich things.
we can zoom the histogram using a filtering on df.col[df.col ><= X, bins=50].
here again, most important are interpreting this data, we can see almost 1500 cars that cost 1000, that probably are the LOWEST PRICE on the page (important, bussiness understanding) then we se a "comon" long tail shi upscaling until 25k app with 700 cars and slowly gettin downscale the count of cars at that prices.
this kind of distribution are nos expected, we search for "normal distribution" as is possible. long tail confuse the model. the ussual trick is to apply logarithm to the price. 
> The logarithm compresses large values. For large numbers, the value of the logarithm is not that large: going from 10 to 1,000 is a big jump, but the increase between their logarithms is not that high. So the logarithm takes very high values and makes them lower.
log 0 doesnt exist and in python treat that as -inf, wich are a problem, even if in df are not 0 values, use np.logip() makes us shure that plus to each value and then takes log.
doing that, in the course ex. the log tail was gone, without any filter we can se how the data are more normalize. more like a bell curve (gauss bell).
> normal distibution: a clear center of the distribution that goes down on both the left and right side.
ofc we have a weird shi at the start but, understanding the bussines should be fine. (probably then we use something as iqr and let outsiders out bc, why we want a model that predict with price that are set by users just to clickbait people, but im just supossing).
models works better when the target variable looks like a normal distrubution. 
for last thing we check the nulls, we see market category have a lot of those, we have to handle it in the next steps.
Pandas attributes and methods:

    df[col].unique() -> return a list of unique values in the series
    df[col].nunique() -> return the number of unique values in the series
    df.isnull().sum() -> return the number of null values in the dataframe

Matplotlib and seaborn methods:

    %matplotlib inline -> assure that plots are displayed in jupyter notebook's cells
    sns.histplot() -> show the histogram of a series

Numpy methods:

    np.log1p() -> apply log transformation to a variable, after adding one to each input value.

Long-tail distributions usually confuse the ML models, so the recommendation is to transform the target variable distribution to a normal one whenever possible.

---

##### 4-validation-framework.md



---
Estimated time on lecture: 

| DAY      | START | FINISH | TOTAL(hr) |
| -------- | ----- | ------ | --------- |
| 05.10.26 | 02.10 | 03.10  |     1     |

Total: 8,45 hrs