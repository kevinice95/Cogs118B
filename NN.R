library("keras")
library("dplyr")
library("ROSE")
library("MLmetrics")
setwd("~/Desktop/UCSD")
data = read.csv("data.csv")
set.seed(42069)
# I stole this.
data <- data %>% dplyr::select(age, c_charge_degree, race, age_cat, score_text,days_b_screening_arrest,
                               sex, priors_count,decile_score,is_recid) %>% 
  filter(days_b_screening_arrest <= 30) %>%
  filter(days_b_screening_arrest >= -30) %>%
  filter(is_recid != -1) %>%
  filter(c_charge_degree != "O") %>%
  filter(score_text != 'N/A') %>% select(-days_b_screening_arrest,is_recid)

X = data %>% select(-score_text, -decile_score) %>% as.data.frame()
y = data$score_text
y <- fastDummies::dummy_cols(y)[,-c(1)]
# We gotta makle dummy variables
X <- cbind(X,fastDummies::dummy_cols(X$race)[,2:6])
X$race <- NULL
X <- cbind(X,fastDummies::dummy_cols(X$age_cat)[,2:3])
X$age_cat <- NULL
X <- cbind(X,fastDummies::dummy_cols(X$c_charge_degree)[,2])
X$c_charge_degree <- NULL
X <- cbind(X,fastDummies::dummy_cols(X$sex)[,2])
X$sex <- NULL
X <- as.matrix(X)
y <- as.matrix(y)
colnames(X) <- c("age","prior_counts","African-American","Asian","Caucasian",
                 "Hispanic","Native-American","25-45years",">45years","isresid",
                 "Felony","Female")

# The next problem is that because most of our observations are in the middle,
# Our neural network can just overfit to it since we have such imbalanced data
## 1144 low risk, 3421 medium risk, 1607 high risk.
## So I'm just going to oversample the other two groups until I get enough
## https://www.analyticsvidhya.com/blog/2016/03/practical-guide-deal-imbalanced-classification-problems/
## for example in that link they show how to deal when you have 2 cateogires, but we have 3.
## So I'm just going to sample until all my variables have the same amount (3421)




model <- keras_model_sequential()
model %>%
  layer_dense(units=5, activation="relu", input_shape = 12,kernel_regularizer = regularizer_l2(l = 0.01)) %>%
  layer_dense(units=3, activation="softmax")

model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy'))

# http://www.milanor.net/blog/cross-validation-for-predictive-analytics-using-r/ for how to do k-cross validation
n_folds <- 5
folds_i <- sample(rep(1:n_folds, length.out = nrow(X)))
for (i in 1:n_folds){
  temp <- which(folds_i==i)
  x_train <- X[-temp,]
  x_test <- X[temp,]
  y_train <- y[-temp,]
  y_test <- y[temp,]
  a <- model %>% fit(x_train, y_train, callbacks = list(
    callback_reduce_lr_on_plateau(monitor = "accuracy", factor = 0.1),
    callback_early_stopping(monitor = "accuracy", min_delta = 0.002,
                            patience = 5, verbose = 0)),batch_size=50,
    epochs=150)
}
# Model seems to be very stable (I know that to test real pradiction I would need)
# to have an additional holdout data not used in k-cross validation. Lets run
# The algorithm on the whole data and then see where it makes the most mistakes
# and in what kind of people.
a <- model %>% fit(X, y, callbacks = list(
  callback_reduce_lr_on_plateau(monitor = "accuracy", factor = 0.001),
  callback_early_stopping(monitor = "accuracy", min_delta = 0.0001,
                          patience = 10, verbose = 0)),batch_size=50,
  epochs=150)
predictions <- predict(model,X)
predictions1 <- predictions
predictions1[cbind(seq_along(1:nrow(predictions1)),max.col(predictions1))] <- 1
predictions1[predictions1!=1] <- 0
# compute accuracy for each column (just to see if there is big misclassifications)
#Low 
print(MLmetrics::Accuracy(predictions1[,1],y[,1]))
print(MLmetrics::Accuracy(predictions1[,2],y[,2]))
print(MLmetrics::Accuracy(predictions1[,3],y[,3]))

## best accuracy for 1, then 2, and worse for 3.




### The other one
y <- as.matrix(data$decile_score)

model1 <- keras_model_sequential()
model1 %>%
  layer_dense(units=5, activation="relu", input_shape = 11) %>%
  layer_dense(units=1, activation='relu')


model1 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

n_folds <- 5
folds_i <- sample(rep(1:n_folds, length.out = nrow(X)))
means <- list()
for (i in 1:n_folds){
  temp = which(folds-i==i)
  x_train <- X[-temp,]
  x_test <- X[temp,]
  y_train <- y[-temp]
  y_test <- y[temp]
  a <- model1 %>% fit(x_train,y_train,epochs=150,batch_size=20)
  predictions <- predict(model1,x_test)
  means[i] <- mean((y_test-predictions)^2)
}
