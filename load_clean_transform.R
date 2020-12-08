#----------------------------Librarys------------------
library(ellipsis)
library(tidyverse)
library(broom)
library(purrr)
library(dummies)
library(scales)
library(GGally)
library(corrplot)
library(gridExtra)
library(grid)
library(h2o)
library(xgboost)
library(ggpubr)
library(caret)
library(DMwR)
library(Metrics)
library(e1071)
library(Ckmeans.1d.dp)
library(usethis)
library(glue)
library(lares)
library(modelplotr)
library(devtools)
#install_github("modelplot/modelplotr")
#install_github("laresbernardo/lares", dependencies = TRUE)
#sessionInfo()
#---------------------Load and check-------------------
data <- read.csv("raw_data/online_shoppers_intention.csv")
head(data)
str(data)
summary(data)

#------------------------------Clean and transformations-------------------------------
nuniques <- as.data.frame(sapply(data, function(x) length(unique(x))))
colnames(nuniques) <- "n_unique"

row.names(nuniques)
#Fiels between 3 and 17 unique values will be transform to factor
for (row in row.names(nuniques)){
    if (nuniques[row,] < 18 & nuniques[row,] > 2)
        data[[row]] <- factor(data[[row]])
}

#--------------------------------Missing data------------------------------
data %>% summarise_all(~ sum(is.na(.)))
na_values<- data[rowSums(is.na(data)) > 0,]
na_values
#14 rows with NA will be remove
df <- drop_na(data)

#Transform Month column to ordered factor to make nice plots
df$Month<- fct_relevel(df$Month, c("Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"))
df$Month<- factor(df$Month, ordered = T)

#------------------------------------------------------------
