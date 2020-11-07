# Databricks notebook source
# MAGIC %md 
# MAGIC 
# MAGIC ### Data analysis using sparklyr, dplyr, countrycode, tidyverse,  mlflow 

# COMMAND ----------

install.packages("tidyverse", type="source")

# COMMAND ----------

# MAGIC %sh 
# MAGIC install.packages("lme4")

# COMMAND ----------

#library(countrycode)
library(tidyverse)

library(SparkR)
library(dplyr)
library(ggplot2)
library(sparklyr)
library(tidyr)
library(mlflow)
install_mlflow()

library(lme4)
library(tidyverse)
library(pscl)
library(parameters)
library(gt)
library(gsubfn)
library(proto)
library(sqldf)
library(RSQLite)

library(usethis)
library(devtools)
library(visdat)
library(skimr)
library(DataExplorer)

# COMMAND ----------

# MAGIC %fs ls /FileStore/tables/ 

# COMMAND ----------

# Load data ---------------------------------------------------------------
#url_survival <- "dbfs:/FileStore/tables/survival_master_dummy.csv"
#url_survival <- "dbfs:/FileStore/tables/survival_master.csv"
#url_merge <- "dbfs:/FileStore/tables/bruno_798_patients_merged_survival_data-4.csv"
#tbl_ncov19 <- read.csv(url_survival)
#tbl_merge <- read.csv(url_merge)

#tbl_symptom_dict <- read.csv("dbfs:/FileStore/tables/symptoms_dictionary.csv")


tbl_ncov19<-read.csv("/dbfs/FileStore/tables/survival_master.csv")

tbl_merge<-read.csv("/dbfs/FileStore/tables/bruno_798_patients_merged_survival_data-4.csv")
tbl_symptom_dict<-read.csv("/dbfs/FileStore/tables/symptoms_dictionary.csv")
glimpse(tbl_ncov19)
glimpse(tbl_merge)




# COMMAND ----------




# COMMAND ----------


