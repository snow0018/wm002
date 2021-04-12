#We will work with the Iris from R

#data(package = 'datasets')
data(iris)#the iris data set are used for illustration of the classification

dim(iris)
str(iris)
levels(iris$Species)
head(iris)

#Exploratory Data Analysis and Visualization
#Plot Iris dataset
#install.packages("caret", dependencies = TRUE)
library("caret")
featurePlot(x = iris[, 1:4], 
            y = iris$Species, 
            plot = "density", # For classification: box, strip, density, pairs or ellipse. For regression, pairs or scatter
            ## Add a key at the top
            auto.key = list(columns = 3)
            )

#Create Training  and Testing Sets
inTrain<-createDataPartition(y=iris$Species, p=0.75, list=FALSE)
training.Iris<-iris[inTrain,]
testing.Iris<-iris[-inTrain,]

#EDA and visualization
dim(training.Iris)
dim(testing.Iris)

#View the training Iris data
boxplot(training.Iris[, -5], main="Raw Data")

##preProcess option allows to  center and scale the data, this is needed for the Training set
preObj<-preProcess(training.Iris[,-5], method = c("center", "scale"))#center是指数据集中的各项数据减去数据集的均值, scale是数据集中的各项数据减去数据集的均值再除以数据集的标准差
preObjData<-predict(preObj,training.Iris[,-5])
boxplot(preObjData, main="Normalized data" )

#build linear Discriminant Analysis (LDA) model
set.seed(1234)
modelFit<-train(Species~., data=training.Iris, preProcess=c("center", "scale"), method="lda")#"."代表全部特征
modelFit$finalModel

#Evaluating this model
#Predict new data with model fitted，and shows Confusion Matrix and performance metrics
predictions<-predict(modelFit, newdata=testing.Iris)
confusionMatrix(predictions, testing.Iris$Species)

#folds<-createFolds(y=training.Iris$Species, k=10, list=T)#k=10表示将数据分为10份，list = TRUE表示函数将返回一个列表
#sapply(folds, length)#查看每个子数据集的样本数量 
#folds[[1]][1:10] #查看第一个子数据集的前10个元素

#compare several models
kfoldcv <- trainControl(method="cv", number=10)
performance_metric <- "Accuracy"

#Linear Discriminant Analysis (LDA)
lda.iris <- train(Species~., data=iris, method="lda", metric=performance_metric, trControl=kfoldcv,preProcess=c("center", "scale"))

#Classification and Regression Trees (CART)
cart.iris <- train(Species~., data=iris, method="rpart", metric=performance_metric, trControl=kfoldcv,preProcess=c("center", "scale"))

#Support Vector Machines (SVM)
svm.iris <- train(Species~., data=iris, method="svmRadial", metric=performance_metric, trControl=kfoldcv,preProcess=c("center", "scale"))

# Random Forest
rf.iris <- train(Species~., data=iris, method="rf", metric=performance_metric, trControl=kfoldcv,preProcess=c("center", "scale"))

results.iris <- resamples(list(lda=lda.iris, cart=cart.iris,  svm=svm.iris, rf=rf.iris))
summary(results.iris)
dotplot(results.iris)

#Parameter Tuning
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
tunegrid <- expand.grid(mtry=c(1:4))
rf_gridsearch <- train(Species~., data=iris, method="rf", metric=performance_metric, tuneGrid=tunegrid, trControl=control,preProcess=c("center", "scale"))
print(rf_gridsearch)
plot(rf_gridsearch)

