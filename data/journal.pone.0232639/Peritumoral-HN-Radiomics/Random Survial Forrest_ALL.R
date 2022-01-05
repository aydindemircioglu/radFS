#script to predict OS from Munich's HCC CT data
library(survival)
library(survminer)
library(caret)
library(randomForestSRC)
#library(randomSurvivalForest)
library(ggRandomForests)
library(rms)

packageurl <- "http://cran.r-project.org/src/contrib/Archive/ggplot2/ggplot2_0.9.1.tar.gz"
install.packages(packageurl, repos=NULL, type="source")

#Set working directory and load both clinical and radiomics .csv files
setwd("C:/Users/S.Sanduleanu/Desktop/DESIGN Tumor Border Analysis 23072019/Oropharynx/TB 3mm/")
Clinical_training<-read.csv("Clinical_DESIGN.csv", sep=';', dec='.', stringsAsFactors=TRUE)
Clinical_validation<-read.csv("Clinical_BD2DECIDE.csv", sep=';', dec='.', stringsAsFactors=TRUE)
Radiomics_training<-read.csv("Radiomics_DESIGN.csv", sep=';', dec='.')
Radiomics_validation<-read.csv("Radiomics_BD2DECIDE.csv", sep=';', dec='.')

#Remove redundant radiomics columns
Var_Radiomics_Training<-Radiomics_DESIGN[,20:1317]
Var_Radiomics_Validation<-Radiomics_BD2DECIDE[,20:1317]

#ComBat feature harmonization from loading the batch file (scanner type) to removing zero variance features to feature harmonization
batch<-as.numeric(c(Clinical_training$Batch, Clinical_validation$Batch))
dat<-dat[,20:1317]

badCols <- nearZeroVar(dat)
if(length(badCols) > 0) dat <- dat[, -nearZeroVar(dat)]

set.seed(789)
dat <- missForest(dat)
dat <- dat$ximp
dat <- data.frame(dat)

dat = as.data.frame(sapply(dat, as.numeric))
a<-a[1:417, drop=FALSE]

HarmonizedTotal = ComBat(dat=t(dat), batch=batch, mod=NULL, par.prior=FALSE)
HarmonizedTotal<- as.data.frame(t(HarmonizedTotal))

Var_Clinical_Training<-Clinical_training[, c('DoB_E1_C1','Sex_E1_C1','WHO_E1_C1', 'Com_E1_C1', 'Packyears_E1_C1', 'Alco_cosump_daily_E1_C1','Haemgl_diagn_E1_C1', 'CTNMN_E1_C1', 'CTNMT_E1_C1','Ch_treatment', 'ChI_E1_C1', 'Chdos_cis_E1_C1','Ch_compl_cis_E1_C1', 'RTdoseTP_E1_C1', 'RTdoseTN_E1_C1', 'TsiteGen_E1_C1')]
Var_Clinical_Validation<-Clinical_validation[, c('DoB_E1_C1','Sex_E1_C1','WHO_E1_C1', 'Com_E1_C1', 'Packyears_E1_C1', 'Alco_cosump_E1_C1','Clinical_HB_Level_Total', 'CTNMN_E1_C1', 'CTNMT_E1_C1','Ch_treatment', 'Chl_E1_C1', 'Chdos_cis_E1_C1','Ch_compl_cis_E1_C1', 'TsiteGen_E1_C1')]

Var_Clinical_Training <- missForest(Var_Clinical_Training)
Var_Clinical_Training <- Var_Clinical_Training$ximp
Var_Clinical_Training <- data.frame(Var_Clinical_Training)

Var_Clinical_Validation <- missForest(Var_Clinical_Validation)
Var_Clinical_Validation <- Var_Clinical_Validation$ximp
Var_Clinical_Validation <- data.frame(Var_Clinical_Validation)

Var_Radiomics_Training <- HarmonizedTotal [1:145,]
Var_Radiomics_Validation <- HarmonizedTotal [146:194,]

time<-Clinical_training$TimeToDeathOrLastFU
status<-Clinical_training$StatusDeath
Var_Radiomics_Training$time<-time
Var_Radiomics_Training$status<-status

survObjT<- Surv(time, status, type='right')
fmla<-as.formula(paste("Surv(time, status)~ ", paste(colnames(Var_Radiomics_Training),collapse= "+")))
#Var_Clinical_Training$time<-time
#Var_Clinical_Training$status<-status
fit <- rfsrc(formula=fmla, data=Var_Radiomics_Training,  ntree=1000, importance=TRUE)
Importance<-vimp(fit,importance = c("permute"))
#plot.rfsrc(fit)
c<-Importance$importance[order(Importance$importance, decreasing = TRUE)]

set.seed(1234)
covariates <- names(Var_Clinical_Training)
Var_Radiomics_Training$time<-Clinical_training$TimeToDistantMetastasis
Var_Radiomics_Training$status<-Clinical_training$StatusDistantMetastasis

#Keep adding highest importance features until first peak in OOB c-index
CindexVector <- matrix(nrow=10, ncol=2)
for(i in c(1:10)){
  selectedVariables <-names(c[i])
  set.seed(789)
  model_rfsrc <- rfsrc(as.formula(paste("Surv(time, status)~ ", paste(selectedVariables), collapse= "+")), data=Var_Radiomics_Training,  ntree=1000, importance=TRUE)
  CindexVector[i,2] <- as.numeric(rcorr.cens(-model_rfsrc$predicted.oob, survObjT)[1])
  CindexVector[i,1] <- i
}

plot(CindexVector, type=c("l"))
selectedVariables <-names(c[1:5])

fit <- rfsrc(survObjT, Var_Clinical_Training, ntree = 1000,
             mtry = NULL, ytry = NULL,
             nodesize = NULL, nodedepth = NULL,
             splitrule = NULL, nsplit = 10,
             importance = c(FALSE, TRUE, "none", "permute", "random", "anti"),
             block.size = if (importance == "none" || as.character(importance) == "FALSE") NULL
             else 10,
             ensemble = c("all", "oob", "inbag"),
             bootstrap = c("by.root", "by.node", "none", "by.user"),
             samptype = c("swr", "swor"), sampsize = NULL, samp = NULL, membership = FALSE,
             na.action = c("na.omit", "na.impute"), nimpute = 1,
             ntime, cause,
             proximity = FALSE, distance = FALSE, forest.wt = FALSE,
             xvar.wt = NULL, yvar.wt = NULL, split.wt = NULL, case.wt  = NULL,
             forest = TRUE,
             var.used = c(FALSE, "all.trees", "by.tree"),
             split.depth = c(FALSE, "all.trees", "by.tree"),
             seed = NULL,
             do.trace = FALSE,
             statistics = FALSE)

##TRAINING

#survival object training
survObjT <- Surv(TimeToEvent,DeathStatus, type='right')

#random survival forest
set.seed(1234)

fitMultipleProgn <- rfsrc(Surv(time, status)~CTNMN_E1_C1+CTNMT_E1_C1+Haemgl_diagn_E1_C1, data=Var_Clinical_Training, mtry=5, ntree=1000, importance=TRUE)
fitMultipleTreat <- rfsrc(Surv(time, status)~Chdos_cis_E1_C1+Ch_compl_cis_E1_C1+ChI_E1_C1, data=Var_Clinical_Training, mtry=5, ntree=1000, importance=TRUE)
fitMultiple <- rfsrc(Surv(time, status)~Chdos_cis_E1_C1+Sex_E1_C1+Haemgl_diagn_E1_C1+Alco_cosump_daily_E1_C1+TsiteGen_E1_C1+Ch_treatment+Com_E1_C1, data=Var_Clinical_Training, mtry=5, ntree=1000, importance=TRUE)
fit <- rfsrc(Surv(time, status)~Wavelet_HLL_GLCM_invDiffMomNor+Wavelet_HLL_GLCM_invDiffNorm+Wavelet_LHL_IH_medianD+Wavelet_HHL_GLCM_invDiffMomNor, data=Var_Radiomics_Training, mtry=5, ntree=1000, importance=TRUE)

SurvObjV <- Surv(Clinical_validation$TimeToDeathOrLastFU,Clinical_validation$Surv_status, type='right')
timeV<-Clinical_validation$TimeToDeathOrLastFU
statusV<-Clinical_validation$Surv_status
Var_Radiomics_Validation$timeV<-timeV
Var_Radiomics_Validation$statusV<-statusV
predDT <- predict(fit, data=Var_Radiomics_Training,importance="none")
predDV <- predict(fit, newdata=Var_Radiomics_Validation,importance="none")
rcorr.cens(-predDT$predicted,  survObjT)["C Index"]
rcorr.cens(-predDV$predicted,  SurvObjV)["C Index"]

#plot(gg_error(RSF))
#rimp <- sort(RSF$importance, decreasing = T)

medianPSP <- median(RSF$predicted.oob)

ModelStrataT <-as.vector(as.numeric(RSF$predicted.oob > medianPSP))

DT$MS <- ModelStrataT
#survival object for plots etc
gg_data <-  gg_survival(interval = "OS", censor = "C", by ="MS", data = DT)
plot(gg_data)

KaplanMeierCurveT <-survfit(survObjT ~ ModelStrataT, data = DT)

ggsurvplot(
  KaplanMeierCurveT,                     # survfit object with calculated statistics.
  pval = TRUE,             # show p-value of log-rank test.
  xlim = c(0, 50),
  conf.int = TRUE,         # show confidence intervals for 
  # point estimaes of survival curves.
  conf.int.style = "step",  # customize style of confidence intervals
  xlab = "Time in Months",   # customize X axis label.
  break.time.by = 5,     # break X axis in time intervals by 200.
  ggtheme = theme_bw(), # customize plot and risk table with a theme.
  risk.table = "abs_pct",  # absolute number and percentage at risk.
  risk.table.y.text.col = T,# colour risk table text annotations.
  risk.table.y.text = FALSE,# show bars instead of names in text annotations
  # in legend of risk table.
  ncensor.plot = TRUE,      # plot the number of censored subjects at time t
  surv.median.line = "hv",  # add the median survival pointer.
  legend.labs = 
    c("Short Term survivors", "Long Term survivors"),    # change legend labels.
  palette = 
    c("#E7B800", "#2E9FDF") # custom color palettes.
)

#Validation
survObjV <- Surv(OSV,CV)

predV = predict(RSF,  newdata = DV,  importance = "none" )

rcorr.cens(-predV$predicted,  survObjV)["C Index"]

ModelStrataV <-as.vector(as.numeric(predV$predicted > medianPSP))

DV$MS <- ModelStrataV
#survival object for plots etc
gg_dataV <-  gg_survival(interval = "OS", censor = "C", by ="MS", data = DV)
plot(gg_dataV)

KaplanMeierCurveV <-survfit(survObjV ~ ModelStrataV, data = DV)

ggsurvplot(
  KaplanMeierCurveV,                     # survfit object with calculated statistics.
  pval = TRUE,             # show p-value of log-rank test.
  xlim = c(0, 50),
  conf.int = TRUE,         # show confidence intervals for 
  # point estimaes of survival curves.
  conf.int.style = "step",  # customize style of confidence intervals
  xlab = "Time in Months",   # customize X axis label.
  break.time.by = 5,     # break X axis in time intervals by 200.
  ggtheme = theme_bw(), # customize plot and risk table with a theme.
  risk.table = "abs_pct",  # absolute number and percentage at risk.
  risk.table.y.text.col = T,# colour risk table text annotations.
  risk.table.y.text = FALSE,# show bars instead of names in text annotations
  # in legend of risk table.
  ncensor.plot = TRUE,      # plot the number of censored subjects at time t
  surv.median.line = "hv",  # add the median survival pointer.
  legend.labs = 
    c("Short Term survivors", "Long Term survivors"),    # change legend labels.
  palette = 
    c("#E7B800", "#2E9FDF") # custom color palettes.
)
