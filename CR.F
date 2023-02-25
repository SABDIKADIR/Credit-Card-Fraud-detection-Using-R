
### loading library
library(ranger)
library(caret)
library(data.table)
library(readxl)
library(readr)
credit_card_data<-read_csv(file = "C:/Users/indir/OneDrive/Desktop/Rcodes/creditcard.csv", show_col_types = FALSE,col_names= TRUE)
credit_card_data


#Data exploration of credit_card_data
dim(credit_card_data)
head(credit_card_data)
tail(credit_card_data)
table(credit_card_data$Class)
summary(credit_card_data$Amount)
names(credit_card_data)
var(credit_card_data$Amount)
sd(credit_card_data$Amount)


#Data Manipulation- first scale the data frame then remove the time column from data frame
head(credit_card_data)
credit_card_data$Amount=scale(credit_card_data$Amount)
newdata=credit_card_data[,-c(1)]
head(newdata)

#Data modeling
library(caTools)
set.seed(123) # create random number
data_sample = sample.split(newdata$Class,SplitRatio = 0.80)
train_data = subset(newdata, data_sample == TRUE)
test_data = subset(newdata,data_sample == FALSE)
dim(train_data)
dim(test_data)

#Fitting logistic Regression Model
logistic_model = glm(Class~.,test_data,family=binomial())
summary(logistic_model)
call:
  glm(formula = Class~., family= binomial(),data = test_data)
plot(logistic_model)

library("pROC")
lr.predict <- predict(logistic_model,train_data, probability = TRUE)
auc.gbm = roc(test_data$Class,lr.predict,plot = TRUE, col = "blue")

#Plotting a decision tree
library(rpart)
library(rpart.plot)
decisionTree_model <- rpart(Class ~ . , credit_card_data, method = 'class')
predicted_val <- predict(decisionTree_model, credit_card_data, type = 'class')
probability <- predict(decisionTree_model, credit_card_data, type = 'prob')
rpart.plot(decisionTree_model)

#Artificial neutral network
library(neuralnet)
ANN_model =neuralnet (Class~.,train_data,linear.output=FALSE)
plot(ANN_model)
predANN=compute(ANN_model,test_data)
resultANN=predANN$net.result
resultANN=ifelse(resultANN>0.5,1,0)

#Gradient Boosting

library(gbm, quietly=TRUE)
system.time(
  model_gbm <- gbm(Class ~ .
                   , distribution = "bernoulli"
                   , data = rbind(train_data, test_data)
                   , n.trees = 500
                   , interaction.depth = 3
                   , n.minobsinnode = 100
                   , shrinkage = 0.01
                   , bag.fraction = 0.5
                   , train.fraction = nrow(train_data) / (nrow(train_data) + nrow(test_data))
  )
)
# Determine best iteration based on test data
gbm.iter = gbm.perf(model_gbm, method = "test")
model.influence = relative.influence(model_gbm, n.trees = gbm.iter, sort. = TRUE)
plot(model_gbm)
gbm_test = predict(model_gbm, newdata = test_data, n.trees = gbm.iter)
gbm_auc = roc(test_data$Class, gbm_test, plot = TRUE, col = "red")
print(gbm_auc)

