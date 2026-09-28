# Notes 01 - Introduction to Machine Learning

##### 01-what-is-ml.md

-> identify variables 
-> understand the business
-> an expert of selling cars works similarity as machine learning models. they can determine the price of a car bc they *learned from data, extracted patterns* (like old car or more mileage is less expensive) *and they applied those patters* to define a just price for the car. thats bc they have the expertise on that, machine learning use data and patterns to using maths determine the correct price. basicly, an expert is a model, if an expert can so can a model.
-> ml: we take data and the model extracts patterns from it.
-> features: the variables/characteristics/everything we know about something.
-> target: what we want to predict using the features.
-> the model encapsulates all the patterns from the data. it is a single artifact we can save and use later.
-> make a prediction are post-create a model, take features of a new car that for example we wanted to sell, put on the model and it will return a price based on the features we give it (make the prediction from the learned patterns). that not mean are the exact price but, based on data, more or less.
> *"To summarize: machine learning is a process of extracting patterns from data. The data consists of features - information about the object - and the target - what we want to predict. The output of machine learning is a model. To use it, we take the features of a new object, put them into the model, and get predictions of the target."*

---

##### 02-ml-vs-rules.md

-> basicly exist a "primitive" way to clasify things, the rule-based system, like if-else statements. using the email example if the mail was sended by xxx@nig.com determines are a spam and also if include the word money and so on, if we set that in like python script, two big errors are commited, 1. the spam mails change a lot in little time, so that list of if-else statemente would change and increse faaast and 2. if a mail from my mother with the word money for something importantn JUST for the word wouls be sended to smap wheras not be an spam. how we solved that? MACHINE LEARNING, its ok start with rulebased quesions, are the base to create features to the model but the big diference are we use the output (spam or not spam) as part of the training to determines as better way how much (%) are spam or not.
-> We take the data, define, create and calclate features, and train the model to be used. 
-> the predictions are probabilities, so we need a bare to deterime when do this or that with that predict.

-> if the feature have 2 states, we can convert them onto binary feature, true or false. Easy (this are not here but xd) for use models like lr if the features work with.

---

##### 03-supervised_ml.md
-> in supervised model we show all the features to reach the target, for that reason are supervised. show examples that what are spam or not for example based on features or this car with these features have this price. we teach it by showing examples.
-> feature matrix (X) where the rows are observations (ex. one row per email) and columns are the features.
-> target variable (y) it is a vector where for each row of X it contains the answer (ex. 1 if spam, 0 if not). For each row of X theres a value in y.
-> the model (g) its a functions that takes X as input and produces something that is aprox close to y. g(X)≈ y.
-> trainings are the process to find g looking the features and coming up with the function.
-> depends of what g output and what the target variable are looks a like are differents machine learning supervised types.
	-> regression: return a NUMBER. (ex.car price)
	-> classification: return a CATEGORY. (ex. spam detection). there are 2 types in this category:
		-> binary classification: just two categories, the target is 0 or 1 and g output a probability between 0 and 1. 
		-> multiclass classification: more than two categories.
	-> ranking: return a ordered list of items responding the answer of, wich item should go first, second, third? for ex.. optimized for that. "the output is the top scores associated with corresponding items. It is applied in recommender systems."

---

##### 04-crisp-dm.md
-> crisp dm is a mthodology wich describes the entire process from understanding the problem to deployment.

![The CRISP-DM process diagram](https://github.com/DataTalksClub/machine-learning-zoomcamp/raw/main/01-intro/images/04-crisp-dm-02-process-diagram-imagegen-pilot.jpg)

1. business understanding: 
	-> define a mesurable goal for a problem to solve. KPI is clave, otherwise how do we lated say erather the project was successful?
	-> undestand the impact of the project.
	-> do we need ml? maybe rule-based system should work, try to not use more than the project need.
2. data understanding:
	-> do we have the data? is it good?
	-> analyze data sources and decide if more data is required.
	-> understand from where the data came and how it was taken. (does this data source really work?)
	-> is this data reliable? maybe are errors on the sampling so a manual check should be good.
	-> is the data large enough?
	-> in this step we can learn more about the business so we can go back to the step 1 and clarify some things.
3. data preparation: 
	-> transform data into a table so we can put into ml model.
	-> clean data, remove noise, appply pipelines.
	-> the output of this step should be X and y from previous lesson.
4. modeling: 
	-> to select the best model, use the validation set.
	->train various ml modles and choose the best one, based on results of this step decide if it is required to add new features or fix data issues.
5. evaluation:
	-> validate that the goal is reached.
	-> solves the business problem determined in step 1?
	-> have we reached the goal? did out metrics improve? if we reduced spam by 30% instaed of 50%, is 30% good enough? maube the proyect are not achievable? iterate.
6. deployment:
	-> roll out to production to all the users.
	-> lately evaluation and deployment happen together (online evaluation, first 5% of users and if it works to all the rest)
	-> ITERATE: 
	-> do something very simple on the first iteration, quickly move thourgh all the spets, evaluate, deploy, learn from the procees, then go back to step 1 and make the model a bit more complex, two or three silly iterations like this dont waste a lot of time and should quickly what your working on is useful.
	1. start simple
	2. learn from the feedback
	3. improve

---

##### 05-model-selection.md
-> we separate the data into 2 datasets instaed of have just 1. train and validation, the model is fitted with "train" data and it us used to predict the y values of the validation feature matrix, then the predictar y values are compared with the actual y values. but this give us a multiple comparisons problem, just by chance one model can be lucky and obtain good predictions because all of them are probabilistic, so we set thre datasets and we use this next recipe to reach the best model:
1. split datasets in training, validation, and test, 60%, 20% and 20% respectivily for ex.
2. train the models with "train 60%"
3. evaluate the models with "validation 20%"
4. select the best model
5. apply the best one to the "test 20%" dataset
6. compare the performance metris of validation and test
after step 4 we can  merge the train and validation dataset, refit the model with the best one option and try on test dataset to fit a little bit better performance and not waste data. 

---

##### 06-enviroment.md
-> this module its focused on set the enviroment to work. personally i feel more comfy using colab but, as a challengue id try to use github codespaces.

---

##### 07-numpy.md
'np.zeros(num)' ; 'np.ones(num)' -> creates an array filled of 0's or 1's with an imput as sized of the array.
'np.full(num, num2)' -> create an array with len num and filled of num2.
'np.array(lst)' -> transform a list into array with the list as an argument.
'arr[i]' access to an element of the array by index. like python, with = we can changue de value of that element in the array.
'np.linespace(inf, sup, len)' create an array of a size len filled with numbers equal separated from inf to sup inputs.
with 'np.zeros((rows, col))' we can also create two-dimensional arrays(matrix) with another input. wich each ones means how much rows and cols have to had the array. That nums have to be as an tuple.
we can create a two-dimensional array like raw 'np.array([1,2,3],[4,5,6],[7,8,9])'
To access to an element we can use index but as x-y cords, arr[0,1], 0 means row and 1 means col.and as the same that bofore, using this access element we can changue the value on that coords. if we pass only one index, return the entire row.
we can rewrite the entire row given one index and a new list/vector to that row. to do the same in the columns we have to set ':' as a first input. 'n[:,1]
'np.random.rand(row,col) -> give us an array with random numbers between 0 and 1 with uniform distribution.
if before that we set the seed with 'np.random.seed(num)' we can psudorandomize the generation, those are affected by an algorithm, with that every wich have set the same seed have to have same random arr.
with np.random.randn(x,y) we can get a random normal distribution.
We can multiply the arrays by a int or float, this multiplate EACH element by that num , just like linear algebra.
'np.random.randint(low=0, high=100, size=(5,2))' randint create an array just with integers.
at the same as multiply i xplain before, we can do element-wise operations to each array, 
As the same that mul, we can multiply and/or sum to each element of an array just using 'arr + num', 'arr * num'. EVERY ELEMENT GET MULTIPLIED OR SUMMED BY THAT NUM.
'np.arange(num)' create an array from 0 to num with len num+1. Also we can divide, substract and everything else we want even chain ops: 'b = (10 + (a * 2)) ** 2 /100' this op is applied element by element.
Also we can compute elemnt-wise between two arrays element by element by index, should be every op that we mentioned before.
Comparision ops are also element-wise. compare 1 by 1 element and return a true or false to the statement. 'a > 2', 'b < a'.
if we use 'a[a > b]' we can select the elements from a wich the statement its true. we can look at all the elements that satisfy a condition.
Also we have the summarizing ops, instaef of element-wise (wich are one by one) those one returns just a number. Ex. 'a.min()' & 'a.max()' the min and max values; 'a.sum()' compute the sum of each element; 'a.mean()' the avg; 'a.std()' the dtandard deviation. All of those works for one and two dimensional arrays. 
There are much more but this works as an introduction to numpy, depend or what we wanted to do are what we used, and tbh its kinda impossible to memorize all this but if we need to do something i can search on google if numpy have a way to do it instaed of think in the standart python way to do it (loops and so on), the goal of use this kind of library, for my, isnt memorize all but search and try to use their functions as many and better as possible, ex. find minimal number in each row, sort those, etc, are specific cases where better are have google close. in the folder i download a cheatcode to check if i need something.; also this two link should be usefulls.
https://mlbookcamp.com/article/numpy
https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/appendix-c-numpy.ipynb

---

##### 08-


---
Estimated time on lecture: 

| DAY      | START | FINISH | TOTAL(hr) |
| -------- | ----- | ------ | --------- |
| 23.09.26 | 00.45 | 01.45  |    1      |
| 24.09.26 | 00.00 | 02.00  |    2      |
| 24.09.26 | 12.45 | 13.30  |   0.75    |
| 28.09.26 | 07.45 | xx.xx  |    x      |



| xx.xx.xx | xx.xx | xx.xx  |    x      |