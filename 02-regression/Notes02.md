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

based of first module, we have to split the dataset onto train, val and test. where we train in train part, validate multiple models with val and the best one we try one time on test. this to try to ignore and esquivate the bias a model. 
Each split have X and y (X-train, y-train) ; (X-val, y-train) ; (X-test, y-test).
first we calcule as train 60%, validate 20% and test20%, to that we multiply n as len(df) with 0,2 each rounding with int. then train are n-val - n_test bc we are rounding maybe aome numbs more or less.
now we have the number of each part, we separate them from the df using iloc[:] to use range on it.
do that have a prolem, as we see on .head(), the df are ordered by 'Make', so we ussually shuffle the dataset to bc we want to make sure t does no affect the split.
to do that we create an array simulate the index of df with the same len and then suffle it with np.random.shuffle(idx), and then we take the % of each part using, again, iloc[] but now with idx[values] using to that, do ing it at this way we shuffle the part of dataset to nos sesgar them. to make this reproducible we set a seed to the random, if not we literally have random resulst always xd.
after that, on each df the index are shuffle, like in random order, we dont really need to know what was the og index of each row, so we reset it on each df using .reset_index(drop=True)
then we separate the results of those datasets to get the y part of g(X) = y using log1p on the case of cars bc the long tail on distribution and then df_train.msrp.values) and then delet it msrp columns from each df. we deletir bc we want to make shure we dont bias the model by without our fault train with the results hahaha.

---

##### 05-linear-regression-simple.md

linear regression is the model for problems where we predict a number, so the output of this model is a number too. 
from the formula g(X)≈y ; g es our linear regression model, X are the feature matrix and y our target.
if we take just three columns as engine horesepower, city miles per gallon and popularity, each row will have a vector's feature; xi = [x, y, z].
the linear regression formula is: 
$$g(x_i) = w_0 + w_1 \cdot x_{i1} + w_2 \cdot x_{i2} + w_3 \cdot x_{i3}$$
w0 are the prediction without know anythig, probably the average.
then we have each wj for xi, that are the multipliers that dermine how much importance or influence each feature has on the final prediction. 
$g(x_i) = w_0 + \sum_{j=1}^{n} w_j \cdot x_i$
for now we dont know how get the weights. but the def to create linear regresion from xi are. this start at 20 and for each feature adds the feature multiplied by its weight:
xi = [453, 11, 86]
w0 = 7.17
w = [0.01, 0.04, 0.002]
def linear_regression(xi):
    n = len(xi)
    pred = w0
    for j in range(n):
        pred = pred + w[j] * xi[j]
    return pred
> The formula says the sum goes from 1 to n, but in Python indexes start at 0, so our loop goes from 0 to n-1. That's the whole implementation - it simply adds up wj times xij for every element of the feature vector.
> To make sense of this number, let's look at the parts it is made of:
> 7.17 + 453·0.01 + 11·0.04 + 86·0.002 = 12.312
>     We start with the bias term, 7.17. This is what we predict for an average car if we don't know anything about it.
>     Then we know the car has 453 horsepower. Each horsepower adds 0.01 to the prediction: if the car had only one horsepower, the price would be higher by 0.01, and for 100 horsepower it's higher by 1. The more horsepower the engine has, the more expensive the car becomes - that's logical.
>     The next feature is miles per gallon in the city, with weight 0.04. The more fuel the car consumes in the city, the more expensive it is. This can make sense: cars that consume more fuel are probably fancier.
>     The last one is popularity, the number of mentions on Twitter, with a pretty low weight of 0.002. It doesn't seem to affect the price much: a lot of people would need to mention a car for this term to matter.
our model predict logarithm of the price bc the distribution, to undo the log we need the exponent using np.expm1(num). so np.expm1 is the inverse of np.log1p.

---

##### 06 linear-regression-vector.md

the previous lesson implement linear regression for one car, now we do it for the entire df using first a dor product for one car, then to a matrix-vector multiplication that produces prediction for all the cars at once.
the sum of the previous furmula are just dot product.
$$g(x_i) = w_0 + x_i^T w$$
the T are traspose, just flipt this column vector onto its side so it becomes a row. with that we can check a compressed version of the formula and, applieable on our code.
> To combine the bias and features into one clean equation, we use a clever mathematical trick by adding a fictional feature of 1 to the beginning of every car's data vector and making the bias the first item in the weight vector. Because any number multiplied by 1 stays the same, doing a standard dot product of these two new lists automatically multiplies the bias by 1 while keeping all other feature-weight multiplications intact. This eliminates the need to handle the bias separately, allowing the entire linear regression formula to be written as a single, streamlined vector multiplication.
after "simplified" (dicen) the formula, do it for all cars:
X^(mx(n+1))
1[1 , x11 , ... , x1n] [w0]
2[1 , x21 , ... , x2n] [w1]
3[1 , .............. ] [..]
m[1 , xm1 , ... , xmn] [wn]
X have m rowns and n+1 columns, we need to take each row of X, do the dot prouct of that row with the vector of w and that give us the pred of tat car. doing it for all th rows give us the vector of predictions y. so now the formula is Xw ≈ y. we can take thre rows, pretend 1 to each feature vector and put them onto a matrix, and then multiplied by w vector. tadaaa, linear regression for complete data.
> To summarize how we got here: we started with a for loop for one car, recognized that it is a dot product between the vector of features and the vector of weights, made it even shorter by adding the fictional feature - the virtual 1 - to the feature vector, and finally generalized it to a complete feature matrix, where linear regression is nothing else but a matrix-vector multiplication.
> The formula of linear regression can be synthesized with the dot product between features and weights. The feature vector includes the bias term with an x value of one, such as w 0 x i 0 ,   w h e r e   x i 0 = 1   f o r   w 0 . When all the records are included, the linear regression can be calculated with the dot product between feature matrix and vector of weights, obtaining the y vector of predictions.

---

###### 07-linear-regression-training.md

where the weights come from? idealy Xw are eqauls y, so we can solve this systemm for w (Xw = y), bc our X matrix isnt square, we cannot solve it for X, m are much larger than n+1.
We aproximate the solution using the Gram Matrix. multipliyng each side by transpose of X.
XᵀXw = Xᵀy
gram matrix are square usually, so have inverse, if not we handle it later.
(XᵀX)⁻¹XᵀXw = (XᵀX)⁻¹Xᵀy
w = (XᵀX)⁻¹Xᵀy <- this is the normal equation to aprox w.
the exact solution for w doesnt exist, this is the closest way possible to get it.
Steps on python:
1. ones = np.ones(X.shape[0])
2. X = np.column_stack([ones, X])
3. XTX = X.T.dot(X)
4. XTX_inv = np.linalg.inv(XTX)
5. XTX.dot(XTX_inv).round(1) #we do this to check if the mul of Gram matrix its inverse, getting I.
6. w_full = XTX_inv.dot(X.T).dot(y)
> The np.column_stack function stacks the vector of ones and the feature matrix together as columns, so the ones become the first column of X.
the w_full have all w, firs one is the bias term, rest are the feature weights. we can split them like w0 = w_full[0] ; w = w_full[1:]
>Note that the weights are negative here. In our earlier examples the weights were positive: more horsepower meant a higher price. A negative weight means the opposite - as the feature value grows, the predicted price goes down. If this feature were the age of a car, that would make perfect sense: the older the car, the lower the price.
> Normal Equation: w = ( X T X ) − 1 X T y ; Where: X T X is the Gram Matrix

---

##### 08-baseline-model.md

the baseline model are the first model where we use just the numerical features, its quality is the reference point we will try to improve on. 
we kno LN (since now bc que paja write linear regression so many times) use just numbers, so. we pick those one and put on a list called base. with that we get a subset of those columns from the df, using df_train[base].values.
in the base line model the ideal are handle the nan with mean but, to keep the process simple we filled with zeros.
then we can train our model (search the w values) and then train y suming w0 and dot product of X_train with w given.
To compare the how good the preds are, we plot them in histogram, using sns.histplot().
following the ex. we can see our baseline model "failed" given us proces smaller than the actual one.
> Just by looking at this chart we suspect the model is not ideal, but that is not an objective way to say it. And when we start improving the model, we want to be sure that it indeed improved. For that we need a single number that quantifies how good or bad the model is. That is exactly what we will do in the next unit with RMSE.

---

##### 9-rmse.md

root mean squared error, the unit that indicate us how good or bad a regression model is. bc a chart is not an objective measure.
rmse the square of differences between predicciont and the actual value for each row, and finally take square root:
RMSE = sqrt( (1/m) · Σ (g(xi) - yi)² );
g(xi) is the prediction
yi is the actual value
m is the number of observations in the dataset (i.e. cars)
MSEare the mean squared error, but the root brings the error back to the same unit as the target.
the implementation in python:
def rmse(y, y_pred):
    error = y - y_pred
    se = error ** 2
    mse = se.mean()
    return np.sqrt(mse)
> The RMSE of the baseline model is about 0.76. This single number is much easier to work with than a chart: whenever we change something - add a feature, transform a variable, try a different model - we recompute it. If the number goes down, the model got better.
So a rmse close to 0, are better

---

##### 10-car-price-validation.md

to be shure about the quality of the model, we apply RMSE to validation dataframe. that would not be bias for any chance.
its a good practice have a def like pipeline to treat the data from each part of dataset as the same way, we use this fot rhis case;
def prepare_X(df):
    df_num = df[base]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X
then, we train our model with df_train BUT, we prove that train using df__val and see what the rmse are. in this case are 0,76, close to the train one so, thats "good", the modellearned from the data very good compare to data that never check before.
> The training part only touches the training dataset: we prepare the matrix and learn the weights. The validation part prepares the validation dataset in the same way as the training dataset, then applies the model learned in the previous step, and computes the root mean squared error for that model.

---

##### 11-feature-engineering.md

> Feature engineering is the process of creating new features from the ones we already have. In this unit we create our first new feature - the age of a car - and see it improve the RMSE of our model quite a bit.
in this case, we create the variable "age" substracting the year of data was recolected and the year. to create a new feature called year and check if our model improve.
and then we modify the prepare def written before to calculate the age, put into df and agregated as feature on in feaures sum base and then, treat as beore, for good practices we DONT modify the actual df_train, we create one, copy that and work with that one. 
def prepare_X(df):
    df = df.copy()
    df['age'] = 2017 - df.year
    features = base + ['age']
    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X
as we can see, adding this new feature, the rmse get low, 0,51 vs 0,76 we get before so, the model are better. if we plot them with histogram we can se are much similar, not perfect but similar.

---

##### 12-categorical-variables.md

we want to use the categorical variables into our model but, ml just work with numbers so, we encode categorical variables, for that we can create a "binary" columns, for ex. if we have num of doors, we can create as many column slike num_doors__2, 3, 4, etc, and put a 1 whereas correct and 0 on the oter ones, encoding the categorical variables. 
using pandas and == op, retunr a list of true/false (basicly binary jaja) so then we use a loop for each numbs of door we have:
for v in [2, 3, 4]:
    df_train['num_doors_%s' % v] = (df_train.number_of_doors == v).astype('int')
now, modify the prepare again, and now, train again and check the rmse (brute force).
in this case the improve was almost null, so the doors freature is not useful.
So we try again but with MAKE and year hahah and the rmse are 0,5076, quite better but not that much.  this was one-hot encoding.








---
Estimated time on lecture: 

| DAY      | START | FINISH | TOTAL(hr) |
| -------- | ----- | ------ | --------- |
| 05.10.26 | 02.10 | 03.10  |     1     |
| 05.10.26 | 09.00 | 11.00  |     2     |
| 12.10.26 | 23.00 | 05.00  |     6     |

Total:  hrs