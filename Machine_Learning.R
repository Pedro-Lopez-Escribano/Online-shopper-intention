#---------------------------------Machine Learning!--------------------
# H2O.ai
h2o.init(
    ip = "localhost",
    nthreads = -1,
    max_mem_size = "4g"
)

df1<- df
df1$Month <- as.character(df1$Month) # H2O don't allows Ordered factors
df_h2o<- as.h2o(df1)
df_h2o$Month <- h2o.asfactor(df_h2o$Month)
h2o.describe(df_h2o)
#----------------------------------
#--------------------------------Train/Test split-----------------------------------------------

particiones <- h2o.splitFrame(data = df_h2o, ratios = c(0.8), seed = 33)
train <- h2o.assign(data = particiones[[1]], key = "train")
test <- h2o.assign(data = particiones[[2]], key = "test")

h2o.table(train$Revenue)/h2o.nrow(train$Revenue)
h2o.table(test$Revenue)/h2o.nrow(test$Revenue)
# Really unbalanced classes, this problem will be solved by
# adjust some arguments in the model construction
train$Revenue <- h2o.asfactor(train$Revenue)
test$Revenue <- h2o.asfactor(test$Revenue)

y <- "Revenue"
x <-setdiff(h2o.colnames(df_h2o),y)
#---------------------------------------------------
#-------------------------H2O MODELS------------------------------------
#When we use H2o it is not necessary to standardize or perform dummy transformations
# or use techniques to balance the classes, H2o automatically transforms the categorical variables,
# and adjusting two parameters we can tell it to standardize the numerical variables
# and balance the classes.

model_glm <- h2o.glm(
    y= y, x= x, training_frame= train,validation_frame = test,
    family= "binomial", link= "logit",
    standardize= T, balance_classes= T, 
    lambda_search= T,
    solver= "AUTO", alpha= 0.95,
    seed= 33,
    nfolds= 5, fold_assignment = "Stratified",
    model_id = "model_glm"
)

model_dp <- h2o.deeplearning( y= y, x= x, training_frame= train,validation_frame = test,
                              balance_classes = TRUE, standardize = TRUE,
                              activation = "Tanh", epochs = 1000, epsilon = 0.0003,
                              loss = "CrossEntropy",input_dropout_ratio = 0.3,
                              stopping_rounds = 5, stopping_metric = "logloss",
                              categorical_encoding = "Binary", verbose = T, model_id = "model_dp")



h2o.saveModel(object = model_dp, path = paste(getwd(),"/ML_models", sep = ""))
h2o.saveModel(object = model_glm, path = paste(getwd(),"/ML_models", sep = ""))

#------------------------AUTO_ML-------------------------------------------------

auto_ML<- h2o.automl(x= x, y= y,training_frame = train, validation_frame = test,
                     nfolds = 5, balance_classes = T)

top3_models<-head(as_tibble(auto_ML@leaderboard)[,1:3],3)

# Save top 3 models
mod_ids <- as_tibble(auto_ML@leaderboard$model_id)

for(i in 1:nrow(mod_ids[0:3,])) {
    
   aml1 <- h2o.getModel(auto_ML@leaderboard[i, 1]) # get model object in environment
   h2o.saveModel(object = aml1, path = paste(getwd(),"/ML_models", sep = ""))
    
}

#It is impossible to use the xgboost library with mplotr, since the xgboost model construction is
#using Dmatrix and this class cannot be transformed to dataframe.
#So I'm going to try make the model through the Caret library.
#----------------------------------------------------------------
#--------------------------XGBoost with Caret-----------------------------
#-------------Data preparation---------------

df$Month <- as.numeric(df$Month)
df1<-df
#-------Transform all categorical variables to factor, to allows us convert it to dummies variables
for (col in colnames(df1)){
    if (n_distinct(df1[[col]]) > 2 & n_distinct(df1[[col]]) < 21){
        df1[[col]] <- factor(df1[[col]])
    }
}

df1<- dummy.data.frame(df1, dummy.classes = "factor")
for (col in colnames(df1)){
    if (class(df1[[col]]) == "factor"){
        df1[[col]] <- as.numeric(df1[[col]])
    }
}
#------------------Train/Test split----------------------------
set.seed(33)
validation_index<-createDataPartition(df1$Revenue, p=0.7, list=FALSE)
testSet<-df1[-validation_index,]
trainSet<-df1[validation_index,]

#------------------------------------Looking for best tuning parameters -----------------------------------------
param <-  expand.grid(nrounds = c(50,100),
                      max_depth = c(5, 6, 10, 12), 
                      eta = c(0.3, 0.1, 0.03), gamma = c(0.5, 1),
                      colsample_bytree = c(0.8, 0.9, 1), min_child_weight = 1,
                      subsample = c(0.5,0.8, 1))  
set.seed(33)
xgb_trcontrol = trainControl(
    method = "cv",
    number = 5,
    verboseIter = TRUE,
    returnData = FALSE,
    returnResamp = "all",              # save losses across all models
    classProbs = TRUE,                 # set to TRUE for LogLoss to be computed
    allowParallel = TRUE,
    summaryFunction = mnLogLoss,
    preProcOptions = "scale")    #scale numeric variables

set.seed(33)
fit.xgb <- train(x = as.matrix(select(trainSet,-Revenue)),
                 y = factor(trainSet$Revenue, labels = c("first_class", "second_class")),
                 method="xgbTree",
                 metric="logLoss", trControl=xgb_trcontrol ,tuneGrid=param)



save(fit.xgb, file =  "ML_models/fit.xgb.RData")
#-----------------------------LOAD MODELS----------------------------------------------------

model1 <- h2o.loadModel(path = "ML_models/StackedEnsemble_BestOfFamily_AutoML_20201021_084846")
model2 <- h2o.loadModel(path = "ML_models/GBM_grid__1_AutoML_20201021_084846_model_60")
model3 <- h2o.loadModel(path = "ML_models/GBM_grid__1_AutoML_20201021_084846_model_32")

model_dp <- h2o.loadModel(path = "ML_models/model_dp")
model_glm <- h2o.loadModel(path = "ML_models/model_glm")

load(paste(getwd(),"/ML_models/fit.xgb.RData", sep = ""))
#------------------------------EVALUATION H2o and XGBoost-------------------------
#The metrics that we are interested in evaluating are the following:
#    Logloss
#    AUC
#    Error classifying class 1, that is Revenue == TRUE
#----------------------------------------------------------

eval <- predict(fit.xgb, testSet,type = "prob")[["second_class"]] > 0.5

confusion_matrix_xgb0 <- cbind(testSet$Revenue, eval) %>% 
    data.frame() %>%
    table() %>% confusionMatrix()

models <- c("Stacked Ensemble auto", "GBM0 auto","GBM1 auto","GLM","Deep Learning","XGB")

error_class_1 <- round(c(h2o.confusionMatrix(model1, valid=T)[,3][2],
                  h2o.confusionMatrix(model2, valid=T)[,3][2],
                  h2o.confusionMatrix(model3, valid=T)[,3][2],
                  h2o.confusionMatrix(model_glm, valid=T)[,3][2],
                  h2o.confusionMatrix(model_dp, valid=T)[,3][2],
                  confusion_matrix_xgb0$table[2,1] / 
                      (confusion_matrix_xgb0$table[2,1] + confusion_matrix_xgb0$table[2,2])),4)

logloss_ <- round(c(h2o.logloss(h2o.performance(model1, valid = T)),
              h2o.logloss(h2o.performance(model2, valid = T)),
              h2o.logloss(h2o.performance(model3, valid = T)),
              h2o.logloss(h2o.performance(model_glm, valid = T)),
              h2o.logloss(h2o.performance(model_dp, valid = T)),
              logLoss(testSet$Revenue, predict(fit.xgb, testSet,type = "prob")[["second_class"]])),4)
              
auc_ <- round(c(h2o.auc(h2o.performance(model1, valid = T)),
          h2o.auc(h2o.performance(model2, valid = T)),
          h2o.auc(h2o.performance(model3, valid = T)),
          h2o.auc(h2o.performance(model_glm, valid = T)),
          h2o.auc(h2o.performance(model_dp, valid = T)),
          auc(testSet$Revenue, predict(fit.xgb, testSet,type = "prob")[["second_class"]])),4)


eval_models <- data.frame(models, auc_, logloss_, error_class_1, stringsAsFactors = F) %>%
    arrange(desc(auc_))

eval_models

#--------------------------------------
# The Lares library allows us to use H2O and provides a series of graphs to evaluate the models,
# it seemed quite visual to me and that is why I wanted to include it in the project.
#-----------------H2o with Lares library------------------------

lares_auto<-h2o_automl(df, y = "Revenue", target = "TRUE" , split = 0.8, balance = TRUE, scale = TRUE, max_models = 10,
                       exclude_algos = NULL, project ="Online shoppers intentions", seed = 33,
                       stopping_metric = "AUC", stopping_rounds = 5, no_outliers = T)

lares_auto$plots$metrics

#---------------------------------------Evaluation best models-------------------------------------
best_models <- c("GBM","XGBoost")

error_class_1 <- round(c(h2o.confusionMatrix(lares_auto$model,)[,3][2],
                   confusion_matrix_xgb0$table[2,1] / 
                       (confusion_matrix_xgb0$table[2,1] + confusion_matrix_xgb0$table[2,2])),4)

logloss_ <- round(c(h2o.logloss(h2o.performance(lares_auto$model, xval = T)),
            logLoss(testSet$Revenue, predict(fit.xgb, testSet,type = "prob")[["second_class"]])),4)

auc_ <- round(c(h2o.auc(h2o.performance(lares_auto$model, xval = T)),
        auc(testSet$Revenue, predict(fit.xgb, testSet,type = "prob")[["second_class"]])),4)

eval_best_models <- data.frame(best_models, auc_, error_class_1, logloss_) %>% arrange(desc(auc_))
eval_best_models

#XGBoost is the best one, but I going to apply SMOTE to trait unbalanced classes problem
#--------------------------------------------------------
#Smote

trainSet$Revenue<- as.factor(trainSet$Revenue)
smoted_data <- SMOTE(Revenue ~ ., trainSet, perc.over=100)
smoted_data$Revenue<- as.numeric(smoted_data$Revenue)
prop.table(table(smoted_data$Revenue)) # equal proportions    
#---------------------
# Grid Search
param <-  expand.grid(nrounds = c(50,100),
                      max_depth = c(5, 6, 10, 12), 
                      eta = c(0.3, 0.1, 0.03), gamma = c(0.5, 1),
                      colsample_bytree = c(0.8, 0.9, 1), min_child_weight = 1,
                      subsample = c(0.5,0.8, 1))  
set.seed(33)
xgb_trcontrol = trainControl(
    method = "cv",
    number = 5,
    verboseIter = TRUE,
    returnData = FALSE,
    returnResamp = "all",              # save losses across all models
    classProbs = TRUE,                 # set to TRUE for LogLoss to be computed
    allowParallel = TRUE,
    summaryFunction = mnLogLoss,
    preProcOptions = "scale")          #scale numeric variables

set.seed(33)
smote_fit.xgb <- train(x = as.matrix(select(smoted_data,-Revenue)),
                       y = factor(smoted_data$Revenue, labels = c("first_class", "second_class")),
                       method="xgbTree",
                       metric="logLoss", trControl=xgb_trcontrol ,tuneGrid=param)


save(smote_fit.xgb,file =  "ML_models/smote_fit.xgb.RData")
#-----------------------------------------------------------
#--------------------Evaluation XGBoost vs smote XGBoost--------------------------

load(paste(getwd(),"/ML_models/smote_fit.xgb.RData", sep = ""))

eval <- predict(smote_fit.xgb, testSet,type = "prob")[["second_class"]] > 0.5

confusion_matrix_xgb1 <- cbind(testSet$Revenue,eval) %>% 
    data.frame() %>%
    table() %>% confusionMatrix()

models <- c("XGB", "XGB Smote")

error_class_1 <- round(c(confusion_matrix_xgb0$table[2,1] / 
                             (confusion_matrix_xgb0$table[2,1] + confusion_matrix_xgb0$table[2,2]),
                         confusion_matrix_xgb1$table[2,1] / 
                             (confusion_matrix_xgb1$table[2,1] + confusion_matrix_xgb1$table[2,2])),4)

logloss_ <- round(c(logLoss(testSet$Revenue, predict(fit.xgb, testSet,type = "prob")[["second_class"]]),
                    logLoss(testSet$Revenue, predict(smote_fit.xgb, testSet,type = "prob")[["second_class"]])),4)

auc_ <- round(c(auc(testSet$Revenue, predict(fit.xgb, testSet,type = "prob")[["second_class"]]),
                auc(testSet$Revenue, predict(smote_fit.xgb, testSet,type = "prob")[["second_class"]])),4)

precision_ <- round(c(precision(testSet$Revenue,predict(fit.xgb, testSet,type = "prob")[["second_class"]] > 0.5 ),
                     precision(testSet$Revenue,predict(smote_fit.xgb, testSet,type = "prob")[["second_class"]] > 0.5)),4)

recall_ <- round(c(recall(testSet$Revenue,predict(fit.xgb, testSet,type = "prob")[["second_class"]] > 0.5 ),
                   recall(testSet$Revenue,predict(smote_fit.xgb, testSet,type = "prob")[["second_class"]] > 0.5 )),4) 


eval_models_xgb <- data.frame(models, auc_, logloss_, error_class_1, precision_, recall_, stringsAsFactors = F) %>%
    arrange(desc(auc_))
eval_models_xgb # XGBoost Smote Wins!!


#----Variable importance XGBOOST Smote---------------
xgb_varImp <- varImp(smote_fit.xgb,scale = F)$importance

xgb_varImp <- rownames_to_column(xgb_varImp,"Variables")
ggplot(head(xgb_varImp,10), aes(x= Overall, y=fct_reorder(Variables, Overall))) + 
    geom_col() + 
        labs(x = "Importance",
             y = "Variable",
             title = "XGBoost Variable Importance")
#----------------------------------
#---------------------------------------Evaluation XGBoost Smote (plots)----------------------------
# Lares plots
mplot_full(tag = testSet$Revenue, 
           score =predict(smote_fit.xgb, testSet,type = "prob")[["second_class"]] , model_name = "XGBoost Smote")

mplot_gain(tag = testSet$Revenue, 
           score =predict(smote_fit.xgb, testSet,type = "prob")[["second_class"]])

mplot_response(tag = testSet$Revenue, 
               score =predict(smote_fit.xgb, testSet,type = "prob")[["second_class"]])

#-----------Compare Confusion Matrix------------------------
mplot_conf(tag = testSet$Revenue, 
           score =predict(smote_fit.xgb, testSet,type = "prob")[["second_class"]],model_name = "XGBoost Smote")


mplot_conf(tag = testSet$Revenue, 
           score =predict(fit.xgb, testSet,type = "prob")[["second_class"]],model_name = "XGBoost")


# Library modelplotr
scores_and_deciles <- prepare_scores_and_ntiles(datasets=list("trainSet","testSet"),
                                                dataset_labels = list("train data","test data"),
                                                models = list("smote_fit.xgb"),  
                                                model_labels = list("xgboost smote"), 
                                                target_column="Revenue") %>% 
    rename(ntl_FALSE = "ntl_first_class",
           ntl_TRUE = "ntl_second_class")

plot_input <- plotting_scope(prepared_input = scores_and_deciles)

plot_cumlift(plot_input, highlight_ntile = 5)
