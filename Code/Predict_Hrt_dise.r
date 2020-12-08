library(caret)
library(pROC)
library(MLeval)


heart_data = read.csv("heart.csv") #Reading the file

#==============|| Pre-Processing Data ||==============

# heart_data[heart_data$sex == 0,]$sex <-"F"
# heart_data[heart_data$sex == 1,]$sex <-"M"
heart_data <- heart_data[!(heart_data$ca == 4 | heart_data$thal == 0),]
heart_data$target <- ifelse(test = heart_data$target == 0, yes = "Healthy", no = "Unhealthy")



heart_data$sex <- as.factor(heart_data$sex)
heart_data$cp <- as.factor(heart_data$cp)
heart_data$fbs <- as.factor(heart_data$fbs)
heart_data$restecg <- as.factor(heart_data$restecg)
heart_data$exang <- as.factor(heart_data$exang)
heart_data$slope<- as.factor(heart_data$slope)
heart_data$ca <- as.factor(heart_data$ca)
# heart_data$thal <- as.factor(heart_data$thal)
heart_data$target <- as.factor(heart_data$target)

#===========|| Update ||===================

# heart_data$ï..age <- as.numeric(heart_data$ï..age)
# heart_data$trestbps <- as.numeric(heart_data$trestbps)
# heart_data$chol <- as.numeric(heart_data$chol)
# heart_data$thalach <- as.numeric(heart_data$thalach)
heart_data[heart_data$thal == 0,]$thal <-"out"
heart_data[heart_data$thal == 1,]$thal <-"normal"
heart_data[heart_data$thal == 2,]$thal <-"fixed_defect"
heart_data[heart_data$thal == 3,]$thal <-"reversable_defect"
heart_data$thal <- as.factor(heart_data$thal)



#==============|| Logistic regression classification ||==============

#Logistic_classification <- glm(target ~ ., data = heart_data, family = "binomial")
set.seed(123) # Setting seed for reproducibility

#Splitting training data into training and testing data in 70:30 ratio.
heart_dataInTrain = createDataPartition(heart_data$target, p = 0.75)[[1]]

#training data set containing 70% of the total values
trainingHeartData = heart_data[ heart_dataInTrain,]

#testing data set containing the remaining 30% of the total values
testingHeartData = heart_data[-heart_dataInTrain,]

#cross-validation using caret's train technique
LogisticModel_2 <- train(target ~ .,
               data = trainingHeartData,
               # preProc = c("center", "scale"),
               trControl = trainControl(method = "cv", number = 10,
                                        classProbs = TRUE, savePredictions = T,
                                        summaryFunction = twoClassSummary),
               metric = "ROC",
               method = "naive_bayes")
               # family=binomial(link = "logit"))

print(LogisticModel_2$result)

pred <- predict(LogisticModel_2, testingHeartData)
confusionMatrix(pred ,testingHeartData$target)
res <- evalm(LogisticModel_2)
res$roc
