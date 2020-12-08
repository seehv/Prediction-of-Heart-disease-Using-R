#==============|| Librares Required ||==============

library(caret)
library(pROC)
library(MLeval)
library(dplyr)
library(stringr)
library(ROCR)
library(vtable)
library(dplyr)
library(yardstick)


heart_data = read.csv("heart.csv") #Reading the file

#==============|| Pre-Processing Data ||==============
heart_data$sex <- as.factor(heart_data$sex)
heart_data$cp <- as.factor(heart_data$cp)
heart_data$fbs <- as.factor(heart_data$fbs)
heart_data$restecg <- as.factor(heart_data$restecg)
heart_data$exang <- as.factor(heart_data$exang)
heart_data$slope<- as.factor(heart_data$slope)
heart_data$ca <- as.factor(heart_data$ca)
heart_data$target <- ifelse(test = heart_data$target == 0, yes = "Healthy", no = "Unhealthy")
heart_data$target <- as.factor(heart_data$target)

heart_data[heart_data$thal == 0,]$thal <-"out"
heart_data[heart_data$thal == 1,]$thal <-"normal"
heart_data[heart_data$thal == 2,]$thal <-"fixed_defect"
heart_data[heart_data$thal == 3,]$thal <-"reversable_defect"
heart_data$thal <- as.factor(heart_data$thal)

label <- data.frame(
  slope = "the slope of the peak exercise ST segment",
  thal = "results of thallium stress test measuring blood flow to the heart",
  trestbps = "resting blood pressure (in mm Hg on admission to the hospital)",
  cp = "chest pain type",
  ca = "number of major vessels colored by flourosopy",
  fbs = "fasting blood sugar > 120 mg/dl",
  restecg = "resting electrocardiographic results",
  chol = "serum cholestoral in mg/dl",
  oldpeak = "ST depression induced by exercise relative to rest, a measure of abnormality in electrocardiograms",
  sex = "sex of the patient",
  ï..age = "age in years", max_heart_rate_achieved = "maximum heart rate achieved (beats per minute)",
  exang = "exercise induced angina",
  target = "Whether or not a patient has heart disease"
)

vtable::vtable(heart_data, labels = label, factor.limit = 0)

#==============|| Data visuvalization -- Uncomment of need to be used ||==============
#==============|| visualization of Numeric and integer features ||==============
# 
# plot_box <- function(heart_data, cols, col_x = "target") {
#   for (col in cols) {
#     p <- ggplot(heart_data, aes(x = .data[[col_x]], y = .data[[col]], fill = .data[[col_x]])) +
#       geom_boxplot(show.legend = FALSE) +
#       scale_fill_manual(values = c("Unhealthy" = "red", "Healthy" = "green"), aesthetics = "fill") +
#       labs(
#         x = "Heart disease present", y = str_c(col),
#         title = str_c("Box plot of", col, "vs", col_x, sep = " "),
#         caption = "Box Plot"
#       ) +
#       theme(
#         axis.text.x = element_text(face = "bold"),
#         axis.title.y = element_text(size = 12, face = "bold"),
#         axis.title.x = element_text(size = 12, face = "bold")
#       )
#     
#     print(p)
#   }
# }
# 
# num_cols <-
#   heart_data %>%
#   select_if(is.numeric) %>%
#   colnames()
# 
# plot_box(heart_data, num_cols)
# 
# #==============|| visualization of Numeric and integer features ||==============
# 
# plot_bars <- function(heart_data, cat_cols, facet_var) {
#   for (col in cat_cols) {
#     p <- ggplot(heart_data, aes(x = .data[[col]], fill = .data[[col]])) +
#       geom_bar(show.legend = F) +
#       labs(
#         x = col, y = "Number of patients",
#         title = str_c("Bar plot of", col, "for heart disease", sep = " ")
#       ) +
#       facet_wrap(vars({{ facet_var }}), scales = "free_y") +
#       theme(
#         axis.title.y = element_text(size = 12, face = "bold"),
#         axis.title.x = element_text(size = 12, face = "bold"),
#         axis.text.x = element_text(angle = 90, hjust = 1, face = "bold")
#       )
#     
#     print(p)
#   }
# }
# 
# cat_cols <-
#   heart_data %>%
#   select_if(is.factor) %>%
#   colnames()
# 
# cat_cols <- cat_cols[-9] # removing the class label
# 
# plot_bars(heart_data, cat_cols, target)

# set.seed(123) # Setting seed for reproducibility

#==============|| Extra Pre-Processing After Visuvalization ||==============

heart_data <- heart_data[!(heart_data$ca == 4 | heart_data$thal == 0),]

#Splitting training data into training and testing data in 70:30 ratio.
heart_dataInTrain = createDataPartition(heart_data$target, p = 0.75)[[1]]

#training data set containing 70% of the total values
trainingHeartData = heart_data[ heart_dataInTrain,]

#testing data set containing the remaining 30% of the total values
testingHeartData = heart_data[-heart_dataInTrain,]

dummies <- dummyVars("~.", data = trainingHeartData, fullRank = T)
xtrain_dummy <- predict(dummies, newdata = trainingHeartData)
xtrain <- as_tibble(xtrain_dummy)

xtest_dummy <- predict(dummies, newdata = testingHeartData)
xtest <- as_tibble(xtest_dummy)

xtrain <- xtrain %>% dplyr::select(-c(restecg.2, target.Unhealthy, fbs.1, ca.4, thal.normal, thal.out))
# Appending Y to the `xtrainDummy` dataset

trainingHeartData <- bind_cols(xtrain, target = trainingHeartData$target)

# Remove low variance columns on test set

xtest <- xtest %>% dplyr::select(-c(restecg.2,target.Unhealthy, thal.out))

# Appending Y to the `xtestDummy` dataset

testingHeartData <- bind_cols(xtest, target = testingHeartData$target)

rm("xtrain_dummy","xtest_dummy","xtest","xtrain","dummies")

#==============|| Building the model for logistic regression ||==============

LogisticModel <- train(target ~ .,
                       data = trainingHeartData,
                       preProc = c("center", "scale"),
                       trControl = trainControl(method = "cv", number = 10,
                                                classProbs = TRUE, savePredictions = T,
                                                summaryFunction = twoClassSummary),
                       metric = "ROC",
                       method = "glm",
                       family=binomial(link = "logit"))

cat("\n",'ROC value for logistic Regression is: ', LogisticModel$result$ROC, "\n")
pred_LR <- predict(LogisticModel, testingHeartData)
ConfMat <- confusionMatrix(pred_LR ,testingHeartData$target)
accu_clasi <- (ConfMat$table[1,1]+ConfMat$table[2,2])/sum(ConfMat$table)
cat("\n",'Accuracy of for logistic Regression model is: ', accu_clasi, "\n")
ConfMat <-conf_mat(ConfMat$table,pred_LR)
autoplot(ConfMat, type = "heatmap", title= "Logistic Resgression")
res <- evalm(LogisticModel)

#==============|| Building the model for Navie Bayes ||==============

#Navie Bayes using caret's train function.

NB_Model <- train(target ~ .,
                  data = trainingHeartData,
                  preProc = c("center", "scale"),
                  trControl = trainControl(method = "cv", number = 10,
                                           classProbs = TRUE, savePredictions = T,
                                           summaryFunction = twoClassSummary),
                  metric = "ROC",
                  method = "naive_bayes")


cat("\n",'ROC value for Navie Bayes where probability  \n of True is: ', NB_Model$result$ROC[1],"\n", 'and False is: ',NB_Model$result$ROC[2], "\n")
pred_NB <- predict(NB_Model, testingHeartData)
ConfMat <- confusionMatrix(pred_NB ,testingHeartData$target)
accu_clasi <- (ConfMat$table[1,1]+ConfMat$table[2,2])/sum(ConfMat$table)
cat("\n",'Accuracy of for Navie Bayes model is: ', accu_clasi, "\n")
ConfMat <-conf_mat(ConfMat$table,pred_NB)
autoplot(ConfMat, type = "heatmap", title= "Naive Bayes")
res <- evalm(NB_Model)

#==============|| Building the model for Decision Tree ||==============
#Decision Tree using caret's train function.

DT_Model <- train(target ~ .,
                  data = trainingHeartData,
                  preProc = c("center", "scale"),
                  trControl = trainControl(method = "cv", number = 10,
                                           classProbs = TRUE, savePredictions = T,
                                           summaryFunction = twoClassSummary),
                  metric = "ROC",
                  method = "rpart")


cat("\n",'ROC value for Decision Tree for main feature Chest pain level is: ', DT_Model$result$ROC, "\n")
pred_DT <- predict(DT_Model, testingHeartData)
ConfMat <- confusionMatrix(pred_DT ,testingHeartData$target)
accu_clasi <- (ConfMat$table[1,1]+ConfMat$table[2,2])/sum(ConfMat$table)
cat("\n",'Accuracy of for Decision Tree model is: ', accu_clasi, "\n")
ConfMat <-conf_mat(ConfMat$table,pred_DT)
plot_p<- autoplot(ConfMat, type = "heatmap", title= "Decision Tree")
plot(plot_p)

res <- evalm(DT_Model)
