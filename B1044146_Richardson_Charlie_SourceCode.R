library(mlbench)

library(caret)

library(corrplot)

#Read Data

weatherHistory <- read.csv("weatherHistory.csv")

set.seed(7)
#Validate Data

validationIndex <- createDataPartition(weatherHistory$Daily.Summary
                                       , p=0.80, list = FALSE)

validation <- weatherHistory[-validationIndex,]

dataset <- weatherHistory[validationIndex,]

#Summary Of Data

dim(dataset)
sapply(dataset, class)
head(dataset, n=20)
summary(dataset)

"dataset[,1] <- as.numeric(as.character(dataset[,1]))
dataset[,2] <- as.numeric(as.character(dataset[,2]))
dataset[,3] <- as.numeric(as.character(dataset[,3]))
dataset[,12] <- as.numeric(as.character(dataset[,12]))"

par(mfrow=c(2,7))
for(i in c(4,5,6,7,8,9,10,11,12))
  {plot(density(dataset[,i]), main=names(dataset)[i])}

par(mfrow=c(2,7))
for(i in c(4,5,6,7,8,9,10,11,12))
  {hist(dataset[,i], main=names(dataset)[i])}

cor(dataset[,c(4,5,6,7,8,9,10,11)])

print(dataset[,c(4,5,6,7,8,9,10,11)])

cor(dataset[,c(4,5,6,7,8,9,10,11)])
correlations <- cor(dataset[,c(4,5,6,7,8,9,10,11)])
corrplot(correlations, method="circle")

pairs(dataset[,c(4,5,6,7,8,9,10,11)])
correlations <- cor(dataset[,c(4,5,6,7,8,9,10,11)])
corrplot(correlations, method="circle")

# these are all the figures to visualize the data
# next run algorithms 


#Algorithm Baseline

trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"

# find all atrribute data

sapply(lapply(dataset, unique), length)

#LM
set.seed(7)
fit.lm <- train(Humidity~., data=dataset, method="lm", metric=metric,
preProc=c("center", "scale"), trControl=trainControl)

#GLM
set.seed(7)
fit.glm <- train(Humidity~., data=dataset, method="glm", metric=metric,
preProc=c("center", "scale"), trControl=trainControl)

#GLMNET
set.seed(7)
fit.glmnet <- train(Humidity~., data=dataset, method="glmnet", metric=metric,
preProc=c("center", "scale"), trControl=trainControl)

#SVM
set.seed(7)
fit.svm <- train(Humidity~., data=dataset, method="svmRadial", metric=metric,
preProc=c("center", "scale"), trControl=trainControl)

#GRID
set.seed(7)
grid <- expand.grid(.cp=c(0, 0.05, 0.1))
fit.cart <- train(Humidity~., data=dataset, method="rpart", metric=metric,
tuneGrid=grid,
preProc=c("center", "scale"), trControl=trainControl)

#KNN
set.seed(7)
fit.knn <- train(Humidity~., data=dataset, method="knn", metric=metric,
preProc=c("center", "scale"), trControl=trainControl)

#results

results <- resamples(list(LM=fit.lm, GLM=fit.glm, GLMNET=fit.glmnet,
SVM=fit.svm, CART=fit.cart, KNN=fit.knn))

summary(results)
dotplot(results)

#Algorithm Feature Selection

# Remove highly correlated attributes from the dataset

# find attributes that are highly corrected

set.seed(7)

cutoff <- 0.70

correlations <- cor(dataset[,c(3,4,5,6,7,8,9)])

highlyCorrelated <- findCorrelation(correlations, cutoff=cutoff)

for (value in highlyCorrelated) {
  print(names(dataset)[value])
  }

# create a new dataset without highly corrected features

datasetFeatures <- dataset[,-highlyCorrelated]

dim(datasetFeatures)

trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"

# LM
set.seed(7)
fit.lm <- train(Humidity~., data=datasetFeatures, method="lm", metric=metric,
preProc=c("center", "scale"), trControl=trainControl)

# GLM
set.seed(7)
fit.glm <- train(Humidity~., data=datasetFeatures, method="glm",metric=metric, 
preProc=c("center", "scale"), trControl=trainControl)

# GLMNET
set.seed(7)
fit.glmnet <- train(Humidity~., data=datasetFeatures, method="glmnet",metric=metric, 
preProc=c("center", "scale"), trControl=trainControl)

# SVM
set.seed(7)
fit.svm <- train(Humidity~., data=datasetFeatures, method="svmRadial",metric=metric, 
preProc=c("center", "scale"), trControl=trainControl)

# CART
set.seed(7)
grid <- expand.grid(.cp=c(0, 0.05, 0.1))
fit.cart <- train(Humidity~., data=datasetFeatures, method="rpart",
metric=metric, tuneGrid=grid, preProc=c("center", "scale"),
trControl=trainControl)

# KNN
set.seed(7)
fit.knn <- train(Humidity~., data=datasetFeatures, method="knn",
metric=metric, preProc=c("center", "scale"), trControl=trainControl)

# Compare algorithms
feature_results <- resamples(list(LM=fit.lm, GLM=fit.glm,
GLMNET=fit.glmnet, SVM=fit.svm, CART=fit.cart, KNN=fit.knn))

summary(feature_results)
dotplot(feature_results)

print(fit.svm)

trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)

metric <- "RMSE"

set.seed(7)

grid <- expand.grid(.sigma=c(0.025, 0.05, 0.1, 0.15), .C=seq(1, 10,by=1))

fit.svm <- train(Temperature..C.~., data=dataset, method="svmRadial", metric=metric,
tuneGrid=grid,
preProc=c("BoxCox"), trControl=trainControl)

print(fit.svm)
plot(fit.svm)

# Estimate accuracy of ensemble methods

trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"

# Random Forest
set.seed(7)
fit.rf <- train(Temperature..C.~., data=dataset, method="rf", metric=metric,
preProc=c("BoxCox"), trControl=trainControl) 

# Stochastic Gradient Boosting
set.seed(7)
fit.gbm <- train(Temperature..C.~., data=dataset, method="gbm", metric=metric,
preProc=c("BoxCox"), trControl=trainControl, verbose=FALSE)

# Cubist
set.seed(7)
fit.cubist <- train(Temperature..C.~., data=dataset, method="cubist", metric=metric,
preProc=c("BoxCox"), trControl=trainControl)

# Compare 
ensembleResults <- resamples(list(RF=fit.rf, GBM=fit.gbm,
CUBIST=fit.cubist))

summary(ensembleResults)

dotplot(ensembleResults)

# Summarize Accuracy 

print(fit.cubist)

# Tune the cubist algorithm

trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"

set.seed(7)
grid <- expand.grid(.committees=seq(15, 25, by=1), .neighbors=c(3, 5, 7))
tune.cubist <- train(Temperature..C.~., data=dataset, method="cubist",
metric=metric,
preProc=c("BoxCox"), tuneGrid=grid, trControl=trainControl)

print(tune.cubist)

plot(tune.cubist)

# Prepare the data transform and finalize the model

set.seed(7)

x <- dataset[,1:9]

y <- dataset[,10]

preprocessParams <- preProcess(x, method=c("BoxCox"))

transX <- predict(preprocessParams, x)

# train the final model

finalModel <- cubist(x=transX, y=y, committees=18)

summary(finalModel)

# Make predictions 

set.seed(7)

valX <- validation[,1:9]

trans_valX <- predict(preprocessParams, valX)

valY <- validation[,14]

# use final model to make predictions on the validation dataset

predictions <- predict(finalModel, newdata=trans_valX, neighbors=3)

rmse <- RMSE(predictions, valY)

r2 <- R2(predictions, valY)

print(rmse)


