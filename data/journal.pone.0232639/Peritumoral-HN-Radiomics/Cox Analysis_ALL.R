library(survival)
library(survminer)
library(data.table)
library(caret)
library(stats)
library(rms)
library(glmnet)
library(qpcR)
library(randomForestSRC)
library(clusterSim)
library(BiocManager)
# if (!requireNamespace("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# BiocManager::install("sva", version = "3.9")
library(sva)
library(coxed)
library(Rfast)
library(RCurl)
library(jsonlite)

rm(list=ls())
setwd("R:/Simon/BD2DECIDE") #Zet working directory

# token <- fromJSON(getURL("https://bd2decide.unipr.it/api/token", userpwd='maastro@bd2decide.eu:b2017d2f', httpauth = 1L))$token
# datasetURL <- "https://bd2decide.unipr.it/api/openclinica/dataset?study_type=concat_rs_ps&extended=true"
# h <- getURL(datasetURL, userpw = paste(token, ':', sep=''), httpauth = 1L)
# RawClinData <- read.csv(text=h, header=TRUE, sep=",")
# write.csv(RawClinData, file = "openclinica_dataset_october.csv")

RawClinData <- read.csv("openclinica_dataset_october.csv",header = TRUE, sep = ";", quote = "\"", dec = ",", fill = TRUE, comment.char = "")
ClinData <- RawClinData[,c("Patient_ID","fhf_CT_Image","Study_Type","Study_Name","ctn_Tumor_Region",
                           "clinical_Age_at_Diagnosis","clinical_Sex",
                           "ctn_Stage_at_Diagnosis_7Edition","ctn_Stage_at_Diagnosis_8Edition",
                           "Survival_Status", "Survival_Time", "patho_HPV_Status_Final",
                           "risk_Smoker_at_Time_of_Diagnosis")]
ClinData <- ClinData[(ClinData$fhf_CT_Image=="Yes"),]
FilterClinData <- ClinData[(ClinData$ctn_Tumor_Region == "Hypopharynx"),]
FilterClinDataR <- FilterClinData[!(FilterClinData$Study_Type == "Prospective"),]
FilterClinDataP <- FilterClinData[(FilterClinData$Study_Type == "Prospective"),]



RadData <- read.csv("RadSheetCleaned2_forDeliv_clean.csv",header = TRUE, sep = ";", quote = "\"", dec = ".", fill = TRUE, comment.char = "")
RadBrescia <- read.csv("BresciaPats.csv",header = TRUE, sep = ";", quote = "\"", dec = ".", fill = TRUE, comment.char = "")
Mergenames <- colnames(RadData)
RadBrasciaM <- RadBrescia[,Mergenames]
RadData <- rbind(RadData,RadBrasciaM)

Data <- merge(
  ClinData,
  RadData,
  by.x = "Patient_ID",  #columnname of patientID in 'SurvivalData' used for merging
  by.y = "BD2_ID",   #columnname of patientID in 'Radiomicsdata' used for merging
  all = FALSE # use only matching rows
)
# Dataids <- Data$Patient_ID
# Radids <- RadData$BD2_ID
# Clinids <- ClinData$Patient_ID
# write.csv(, file = ".csv")

Data <- Data[order(Data$Study_Name),]

# FData <- Data[(Data$ctn_Tumor_Region == "Hypopharynx"),]
FData <- Data
# HData <- FData[!(FData$patho_HPV_Status_Final == "Positive"),]
HData <- FData
RetroData <- HData[!(HData$Study_Type == "Prospective"),]
ProspData <- HData[(HData$Study_Type == "Prospective"),]
RetroRadData <- RetroData[(length(ClinData)+1):length(RetroData)]
RetroClinData <- RetroData[1:length(ClinData)]
badCols <- nearZeroVar(RetroRadData[,])
RetroRadData <- RetroRadData[,-badCols]

ProspRadData<-ProspData[(length(ClinData)+1):length(ProspData)]
ProspClinData <- ProspData[1:length(ClinData)]
ProspRadData <- ProspRadData[,-badCols]

DesignClin <- read.csv("R:/Simon/BD2DECIDE/Clinical_DESIGN2.csv",header = TRUE, sep = ";", quote = "\"", dec = ",", fill = TRUE, comment.char = "")
DesignData <- read.csv("R:/Simon/BD2DECIDE/DESIGNLog.csv",header = TRUE, sep = ";", quote = "\"", dec = ".", fill = TRUE, comment.char = "")

ExtData <- merge(
  DesignClin,
  DesignData,
  by.x = "StudySubjectID",  #columnname of patientID in 'SurvivalData' used for merging
  by.y = "Patient_ID",   #columnname of patientID in 'Radiomicsdata' used for merging
  all = FALSE # use only matching rows
)

ExtData <- ExtData[!(ExtData$Study_Name == "VUMC"),]

DesignClin <- ExtData[,1:224]
DesignData <- ExtData[,225:3034]

badCols2 <- nearZeroVar(DesignData[,])
DesignData <- DesignData[,-badCols2]

interdata <- intersect( colnames(RetroRadData),  colnames(DesignData))
DesignData <- DesignData[,interdata]
ProspRadData <- ProspRadData[,interdata]
RetroRadData <- RetroRadData[,interdata]

ExtClin <- DesignClin[,c("Study_Name","Survival_Status","Survival_Time")]
ExtHarmRetro <- RetroClinData[,c("Study_Name","Survival_Status","Survival_Time")]



{
CombRadData <- rbind(RetroRadData,ProspRadData)
CombClinData <- rbind(RetroClinData,ProspClinData)


BatchR <-  CombClinData$Study_Name

HarmonizedR<- ComBat(dat=as.matrix(t(CombRadData)), batch = BatchR, par.prior=TRUE)
HarmonizedR_D<- as.data.frame(t(HarmonizedR))
RetroRadData <- HarmonizedR_D[1:687,]
ProspRadData <- HarmonizedR_D[688:830,]
}

{
ExtCombClinData <- rbind(ExtHarmRetro,ExtClin)
ExtCombRadData <- rbind(RetroRadData,DesignData)
# badCols3 <- nearZeroVar(ExtCombRadData[,])
# ExtCombRadData <- ExtCombRadData[,-badCols3]

BatchRExt <-  ExtCombClinData$Study_Name
table(BatchRExt)
BatchRExt <- droplevels(BatchRExt, exclude = c("ULM","BRESCIA","Maastro"))

HarmonizedRExt<- ComBat(dat=as.matrix(t(ExtCombRadData)), batch = BatchRExt, par.prior=TRUE)
HarmonizedR_DExt<- as.data.frame(t(HarmonizedRExt))
# RetroRadData <- HarmonizedR_D[1:687,]
DesignRadData <- HarmonizedR_D[688:958,]
}

{
BatchR <- RetroClinData$Study_Name
BatchR <- droplevels(BatchR, exclude = c("ULM","BRESCIA"))

HarmonizedR<- ComBat(dat=as.matrix(t(RetroRadData)), batch = BatchR, par.prior=TRUE)
HarmonizedR_D<- as.data.frame(t(HarmonizedR))
RetroRadData <- HarmonizedR_D

table(ProspClinData$Study_Name)

AOPProspRadData <- ProspRadData[(ProspClinData$Study_Name == "AOP"),]
AOPRetroHarm <- RetroRadData[(RetroClinData$Study_Name == "AOP"),]
AOPCom <- rbind(AOPProspRadData,AOPRetroHarm)
AOPBatch <- c(rep(1,nrow(AOPProspRadData)),rep(2,nrow(AOPRetroHarm)))

HarmonizedAOP = ComBat(dat=as.matrix(t(AOPCom)), batch=AOPBatch, par.prior=TRUE, ref.batch=2)
HarmonizedAOP_D<- as.data.frame(t(HarmonizedAOP))
AOPProspRadDataH <- HarmonizedAOP_D
AOPProspRadDataH <- AOPProspRadDataH[1:nrow(AOPProspRadData),]

BRESCIAProspRadData <- ProspRadData[(ProspClinData$Study_Name == "BRESCIA"),]
BRESCIARetroHarm <- RetroRadData
BRESCIACom <- rbind(BRESCIAProspRadData,BRESCIARetroHarm)
BRESCIAComB <- RetroClinData$Study_Name
BRESCIAComB <- droplevels(BRESCIAComB, exclude = c("ULM","BRESCIA"))
NumBRESCIAComB <- summary(BRESCIAComB)

# BRESCIABatch <- c(rep(1,nrow(BRESCIAProspRadData)),rep(2,as.numeric(NumBRESCIAComB["AOP"])),rep(2,as.numeric(NumBRESCIAComB["INT"])),rep(2,as.numeric(NumBRESCIAComB["MAASTRO"])),rep(2,as.numeric(NumBRESCIAComB["UDUS"])),rep(2,as.numeric(NumBRESCIAComB["VUMC"])))
BRESCIABatch <- c(rep(1,nrow(BRESCIAProspRadData)),rep(2,nrow(RetroRadData)))

HarmonizedBRESCIA = ComBat(dat=as.matrix(t(BRESCIACom)), batch=BRESCIABatch, par.prior=TRUE, ref.batch=c(2))
HarmonizedBRESCIA_D<- as.data.frame(t(HarmonizedBRESCIA))
BRESCIAProspRadDataH <- HarmonizedBRESCIA_D
BRESCIAProspRadDataH <- BRESCIAProspRadDataH[1:nrow(BRESCIAProspRadData),]


INTProspRadData <- ProspRadData[(ProspClinData$Study_Name == "INT"),]
INTRetroHarm <- RetroRadData[(RetroClinData$Study_Name == "INT"),]
INTCom <- rbind(INTProspRadData,INTRetroHarm)
INTBatch <- c(rep(1,nrow(INTProspRadData)),rep(2,nrow(INTRetroHarm)))

HarmonizedINT = ComBat(dat=as.matrix(t(INTCom)), batch=INTBatch, par.prior=TRUE, ref.batch=2)
HarmonizedINT_D<- as.data.frame(t(HarmonizedINT))
INTProspRadDataH <- HarmonizedINT_D
INTProspRadDataH <- INTProspRadDataH[1:nrow(INTProspRadData),]

MAASTROProspRadData <- ProspRadData[(ProspClinData$Study_Name == "MAASTRO"),]
MAASTRORetroHarm <- RetroRadData[(RetroClinData$Study_Name == "MAASTRO"),]
MAASTROCom <- rbind(MAASTROProspRadData,MAASTRORetroHarm)
MAASTROBatch <- c(rep(1,nrow(MAASTROProspRadData)),rep(2,nrow(MAASTRORetroHarm)))

HarmonizedMAASTRO = ComBat(dat=as.matrix(t(MAASTROCom)), batch=MAASTROBatch, par.prior=TRUE, ref.batch=2)
HarmonizedMAASTRO_D<- as.data.frame(t(HarmonizedMAASTRO))
MAASTROProspRadDataH <- HarmonizedMAASTRO_D
MAASTROProspRadDataH <- MAASTROProspRadDataH[1:nrow(MAASTROProspRadData),]

ULMProspRadData <- ProspRadData[(ProspClinData$Study_Name == "ULM"),]
ULMRetroHarm <- RetroRadData
ULMCom <- rbind(ULMProspRadData,ULMRetroHarm)
ULMComB <- RetroClinData$Study_Name
ULMComB <- droplevels(ULMComB, exclude = c("ULM","BRESCIA"))
NumULMComB <- summary(ULMComB)

# ,rep(2,as.numeric(NumULMComB["AOP"]))
ULMBatch <- c(rep(1,nrow(ULMProspRadData)),rep(2,nrow(RetroRadData)))

HarmonizedULM = ComBat(dat=as.matrix(t(ULMCom)), batch=ULMBatch, par.prior=TRUE, ref.batch=c(2))
HarmonizedULM_D<- as.data.frame(t(HarmonizedULM))
ULMProspRadDataH <- HarmonizedULM_D
ULMProspRadDataH <- ULMProspRadDataH[1:nrow(ULMProspRadData),]

#
test <- ProspRadData
ProspRadData <- rbind(AOPProspRadDataH,BRESCIAProspRadDataH,INTProspRadDataH,MAASTROProspRadDataH,ULMProspRadDataH)
# ProspRadData <- rbind(BRESCIAProspRadDataH,INTProspRadDataH,MAASTROProspRadDataH,ULMProspRadDataH)
# ProspRadData <- rbind(AOPProspRadDataH,BRESCIAProspRadDataH,INTProspRadDataH)
# ProspRadData <- rbind(AOPProspRadDataH,INTProspRadDataH)
}

AVLProspRadData <- DesignData[(ExtClin$Study_Name == "AVL"),]
AVLRetroHarm <- RetroRadData
AVLCom <- rbind(AVLProspRadData,AVLRetroHarm)
AVLComB <- RetroClinData$Study_Name
AVLComB <- droplevels(AVLComB, exclude = c("AVL","BRESCIA"))
NumAVLComB <- summary(AVLComB)

# ,rep(2,as.numeric(NumAVLComB["AOP"]))
AVLBatch <- c(rep(1,nrow(AVLProspRadData)),rep(2,nrow(RetroRadData)))

HarmonizedAVL = ComBat(dat=as.matrix(t(AVLCom)), batch=AVLBatch, par.prior=TRUE, ref.batch=c(2))
HarmonizedAVL_D<- as.data.frame(t(HarmonizedAVL))
AVLProspRadDataH <- HarmonizedAVL_D
AVLProspRadDataH <- AVLProspRadDataH[1:nrow(AVLProspRadData),]

UMCProspRadData <- DesignData[(ExtClin$Study_Name == "UMC"),]
UMCRetroHarm <- RetroRadData
UMCCom <- rbind(UMCProspRadData,UMCRetroHarm)
UMCComB <- RetroClinData$Study_Name
UMCComB <- droplevels(UMCComB, exclude = c("UMC","BRESCIA"))
NumUMCComB <- summary(UMCComB)

# ,rep(2,as.numeric(NumUMCComB["AOP"]))
UMCBatch <- c(rep(1,nrow(UMCProspRadData)),rep(2,nrow(RetroRadData)))

HarmonizedUMC = ComBat(dat=as.matrix(t(UMCCom)), batch=UMCBatch, par.prior=TRUE, ref.batch=c(2))
HarmonizedUMC_D<- as.data.frame(t(HarmonizedUMC))
UMCProspRadDataH <- HarmonizedUMC_D
UMCProspRadDataH <- UMCProspRadDataH[1:nrow(UMCProspRadData),]

DesignRadData <- rbind(AVLProspRadDataH,UMCProspRadDataH)

scaledR.dat <- scale(RetroRadData)
RetroRadDataS <- data.frame(scaledR.dat)

colm <- colMeans(RetroRadData, na.rm = FALSE, dims = 1)
colsd <-  apply(RetroRadData, 2, sd)

scaled.dat <- scale(ProspRadData,center = colm, scale = colsd)
ProspRadDataS <- data.frame(scaled.dat)

scaled.dat <- scale(DesignRadData,center = colm, scale = colsd)
DesignRadDataS <- data.frame(scaled.dat)

# RetroRadDataNoVol <- RetroRadDataS[,-which(names(RetroRadDataS) %in% c("Shape_volume"))]

# corMatrixV =  cor(RetroRadDataNoVol, RetroRadDataS[,which(names(RetroRadDataS) %in% c("Shape_volume"))])
# highly_vol_correlated_columns <- which(abs(corMatrixV)>0.6)
# RetroRadDataSV <- RetroRadDataNoVol[,-highly_vol_correlated_columns]
# RetroRadDataSV$Shape_volume <- RetroRadDataS$Shape_volume
RetroRadDataSV <- RetroRadDataS

corMatrix =  cor(RetroRadDataSV, y = NULL, use = "complete.obs", method = "spearman")
highly_correlated_columns = findCorrelation(
  corMatrix,
  cutoff = 0.85,
  verbose = FALSE,
  names = FALSE,
  exact = TRUE
)

RetroRadDataSC <- RetroRadDataSV[,-highly_correlated_columns]
RetroDataCom <- cbind(RetroClinData,RetroRadDataSC)

TwoYearSurv <- RetroDataCom$Survival_Time > 730

TotalCI <- vector(mode="numeric",length=100)
TotalNF <- vector(mode="numeric",length=100)
TotalNF2 <- vector(mode="numeric",length=100)
FeatureList <- matrix(data = NA, nrow = 100, ncol = 100)

for (i in(1:100)){
  set.seed(12356789+i)
  Indices <- sample(length(TwoYearSurv))
  IDs <- RetroDataCom$Patient_ID[Indices]
  RetroDataT <- RetroDataCom[RetroDataCom$Patient_ID %in% IDs[1:(length(Indices) * 0.8)],]
  RetroDataV <- RetroDataCom[not(RetroDataCom$Patient_ID %in% IDs[1:(length(Indices) * 0.8)]),]
  
  RetroRadDataT <- RetroDataT[(length(ClinData)+1):length(RetroDataT)]
  RetroRadDataV <- RetroDataV[(length(ClinData)+1):length(RetroDataT)]
  RetroClinDataT <- RetroDataT[1:length(ClinData)]
  RetroClinDataV <- RetroDataV[1:length(ClinData)]
  SurvObjT <- Surv(RetroClinDataT$Survival_Time/30.44,RetroClinDataT$Survival_Status)
  SurvObjV <- Surv(RetroClinDataV$Survival_Time/30.44,RetroClinDataV$Survival_Status)
  
  DT <- RetroRadDataT
  ddist <- datadist(DT)
  DV <- RetroRadDataV
  covariates <- colnames(DT)
  runiv_formulas <-
    sapply(covariates, function(x)
      as.formula(paste('SurvObjT ~', x))) 
  ##create a cox-model for each individual feature
  runiv_models <-
    lapply(runiv_formulas, function(x) {
      coxph(x, data = DT)
    })
  
  ## Calculate significance of each feature using the Wald test
  runiv_results <- lapply(runiv_models, function(x) {
    x <- summary(x)
    p.value <- signif(x$wald["pvalue"], digits = 2)
    beta <- signif(x$coef[1], digits = 2)
    #coeficient beta
    HR <- signif(x$coef[2], digits = 5)
    #exp(beta)
    HR.confint.lower <- signif(x$conf.int[, "lower .95"], 2)
    HR.confint.upper <- signif(x$conf.int[, "upper .95"], 2)
    wald.test <- paste0(signif(x$wald["test"], digits = 2),
                        " (",
                        HR.confint.lower,
                        "-",
                        HR.confint.upper,
                        ")")
    res <- c(beta, HR, wald.test, p.value) #save results
    names(res) <- c("beta", "HR", "wald.test",
                    "p.value")
    return(res)
  })
  rres <- t(as.data.frame(runiv_results, check.names = FALSE))
  runivariate_results <- as.data.frame(rres)
  plist <- as.numeric(levels(runivariate_results$p.value))[runivariate_results$p.value]
  
  qlist <- p.adjust(plist,method = "fdr",n=length(plist))
  rfiltered_uv_results <-
    runivariate_results[qlist < 0.05, ]
  
  # if (nrow(rfiltered_uv_results) < 1) {
  # qlist <- p.adjust(plist,method = "fdr",n=length(plist))
  # rfiltered_uv_results <-
  #   runivariate_results[qlist < 0.05, ]
  # }
  
  # if (nrow(rfiltered_uv_results) < 1) {
  # qlist <- p.adjust(plist,method = "none",n=length(plist))
  # rfiltered_uv_results <-
  #   runivariate_results[qlist < 0.05, ]
  # }
  ##If not enough highly significant features, then only select significant features (p<0.05) 
  # if (nrow(filtered_uv_results) < Nf) {
  #   filtered_uv_results <-
  #     univariate_results[as.numeric(as.character(univariate_results$p.value)) < 0.05/(0.5*ncol(DT)), ]
  # }
  
  ##use the Hazard ratio (HR) to select the top #Nf features for model building  
  ## Extract hazard ratio
  HR <-
    as.numeric(levels(rfiltered_uv_results$HR))[rfiltered_uv_results$HR]
  ## Create modified HR for ranking
  inverseHR <- function(x) {
    if (x < 1) {
      return(1 / x)
    } else{
      return(x)
    }
  }
  IHR <- sapply(HR, inverseHR)
  ## sort results by modified HR
  rsorted_uv_results <-
    sort(IHR, index.return = TRUE, decreasing = TRUE)
  
  features <-
    names(rfiltered_uv_results$beta[rsorted_uv_results$ix])
  
  # AICVector <- matrix(nrow=length(features), ncol=1)
  # for(j in c(1:length(features))){
  #   selectedVariables <-features[1:j]
  #   fmla <- as.formula(paste("SurvObjT ~ ", paste(selectedVariables, collapse= "+")))
  #   coxReg <- coxph(fmla, data = DT, x=TRUE, y=TRUE)
  #   predT = predict(coxReg,  type = "expected")
  #   AICVector[j,1] <- AIC (object = coxReg, k=2)     #as.numeric(rcorr.cens(predT, survObjTraining)["C Index"])[1]
  # }
  # 
  # plot(AICVector, type=c("l"))
  # AICVector
  # NF <- which(AICVector[1:length(AICVector)]==min(AICVector[1:length(AICVector)]))
  loopl <- length(features)
  if (length(features)>35) {
    loopl <- 35
  }
  CIVvector <- vector(mode="numeric",length=loopl)
  for(j in(1:loopl)){
    selectedVariables <-features[1:j]
    fmla <- as.formula(paste("SurvObjT ~ ", paste(selectedVariables, collapse= "+")))
    
    coxReg <- cph(fmla, data = DT, x=TRUE, y=TRUE)
    predictionsT <- survest(coxReg, DT, times = 24)
    medianPSP <- median(predictionsT$surv)
    predictionsV <- survest(coxReg, DV, times = 24, se.fit = TRUE)
    ModelStrataV <-
      as.vector(as.numeric(predictionsV$surv > medianPSP))
    CIVvector[j] <- rcorr.cens(predictionsV$surv, SurvObjV)["C Index"]
  }
  plot(CIVvector, type=c("l"))
  NF <- which(CIVvector[1:length(CIVvector)]==max(CIVvector[1:length(CIVvector)]))
  CIvMax <- max(CIVvector[1:length(CIVvector)])
  DifVector <- diff(CIVvector)
  NF2 <- which(DifVector <= 0)[1]
  
  # fmla <-
  #   as.formula(paste("SurvObjT ~ ", paste((
  #     features[1:NF]
  #   ), collapse = "+")))
  # 
  # nDT = DT[features]
  # coxRegNewT <- cph(fmla, data = nDT, x = TRUE, y = TRUE)
  # predictionsT <- survest(coxRegNewT, nDT, times = 12)
  # medianPSP <- median(predictionsT$surv)
  # ModelStrataT <-
  #   as.vector(as.numeric(predictionsT$surv > medianPSP))
  # CIt <- rcorr.cens(predictionsT$surv, SurvObjT)["C Index"]
  # sdfT <- survdiff(SurvObjT ~ ModelStrataT)
  # PvalT <- 1 - pchisq(sdfT$chisq, length(sdfT$n) - 1)
  # nDV = RetroRadDataV[features]
  # predictionsV <- survest(coxRegNewT, nDV, times = 12)
  # ModelStrataV <-
  #   as.vector(as.numeric(predictionsV$surv > medianPSP))
  # CIv <- rcorr.cens(predictionsV$surv, SurvObjV)["C Index"]
  # sdf <- survdiff(SurvObjV ~ ModelStrataV)
  # PvalV <- 1 - pchisq(sdf$chisq, length(sdf$n) - 1)
  
  TotalNF[i] <- NF
  TotalNF2[i] <- NF2
  TotalCI[i] <- CIvMax
  FeatureList[i,1:NF] <- features[1:NF]
  print(i)
}

MeanNF <- as.numeric(names(sort(table(TotalNF),decreasing=TRUE)[1]))
if (MeanNF == 1) {
  MeanNF <- as.numeric(names(sort(table(TotalNF),decreasing=TRUE)[2]))
}
MeanNF2 <- as.numeric(names(which.max(table(TotalNF2))))
if (MeanNF2 == 1) {
  MeanNF2 <- as.numeric(names(sort(table(TotalNF2),decreasing=TRUE)[2]))
}
NFT <- sum(sort(table(FeatureList),decreasing=TRUE) > 30, na.rm = TRUE)

FeatureListO <- names(sort(table(FeatureList),decreasing=TRUE)[1:MeanNF2])
# FeatureListO <- "Shape_volume"

# WEIGHTED RANK?

TSurvObjT <- Surv(RetroData$Survival_Time/30.44,RetroData$Survival_Status)
fmla <-
  as.formula(paste("TSurvObjT ~ ", paste((
    FeatureListO
  ), collapse = "+")))

# RetroRadDataSC

TDT <- RetroRadDataSV[FeatureListO]
coxRegNewT <- cph(fmla, data = TDT, x = TRUE, y = TRUE)
predictionsT <- survest(coxRegNewT, TDT, times = 24, se.fit = TRUE)
medianPSP <- median(predictionsT$surv)
ModelStrataT <-
  as.vector(as.numeric(predictionsT$surv > medianPSP))
CIt <- rcorr.cens(predictionsT$surv, TSurvObjT)["C Index"]
sdfT <- survdiff(TSurvObjT ~ ModelStrataT)
PvalT <- 1 - pchisq(sdfT$chisq, length(sdfT$n) - 1)

TDV <- ProspRadDataS[FeatureListO]
predictionsV <- survest(coxRegNewT, TDV, times = 24, se.fit = TRUE)
ScoreV <- predictionsV$surv
ModelStrataV <-
  as.vector(as.numeric(predictionsV$surv > medianPSP))
TSurvObjV <- Surv(ProspData$Survival_Time/30.44,ProspData$Survival_Status)
sdfV <- survdiff(TSurvObjV ~ ModelStrataV)
PvalV <- 1 - pchisq(sdfV$chisq, length(sdfV$n) - 1)

# ,"patho_HPV_Status_Final"
DT <- RetroClinData[,c("clinical_Age_at_Diagnosis","clinical_Sex")]
DT$Stage <- as.numeric(RetroClinData$ctn_Stage_at_Diagnosis_8Edition) < 5
DT$SmokeCur <- RetroClinData$risk_Smoker_at_Time_of_Diagnosis =='Current'
DT$SmokeCurFor <- RetroClinData$risk_Smoker_at_Time_of_Diagnosis == 'Current'|RetroClinData$risk_Smoker_at_Time_of_Diagnosis == 'Former'
# DT$RegionOR <- RetroClinData$ctn_Tumor_Region == 'Oropharynx'
# DT$RegionOC <- RetroClinData$ctn_Tumor_Region == 'Oral Cavity'
# DT$RegionHY <- RetroClinData$ctn_Tumor_Region == 'Hypopharynx'
# DT$RegionLA <- RetroClinData$ctn_Tumor_Region == 'Larynx'

covariates <- colnames(DT)
runiv_formulas <-
  sapply(covariates, function(x)
    as.formula(paste('TSurvObjT ~', x))) 
##create a cox-model for each individual feature
runiv_models <-
  lapply(runiv_formulas, function(x) {
    coxph(x, data = DT)
  })

## Calculate significance of each feature using the Wald test
cruniv_results <- lapply(runiv_models, function(x) {
  x <- summary(x)
  p.value <- signif(x$wald["pvalue"], digits = 2)
  beta <- signif(x$coef[1], digits = 2)
  #coeficient beta
  HR <- signif(x$coef[2], digits = 5)
  #exp(beta)
  HR.confint.lower <- signif(x$conf.int[, "lower .95"], 2)
  HR.confint.upper <- signif(x$conf.int[, "upper .95"], 2)
  wald.test <- paste0(signif(x$wald["test"], digits = 2),
                      " (",
                      HR.confint.lower,
                      "-",
                      HR.confint.upper,
                      ")")
  res <- c(beta, HR, wald.test, p.value) #save results
  names(res) <- c("beta", "HR", "wald.test","p.value")
  return(res)
})
# 
rres <- t(as.data.frame(cruniv_results, check.names = FALSE))
runivariate_results <- as.data.frame(rres)
plist <- as.numeric(levels(runivariate_results$p.value))[runivariate_results$p.value]
qlist <- p.adjust(plist,method = "none",n=length(plist))
rfiltered_uv_results <-
  runivariate_results[qlist < 0.05, ]


##use the Hazard ratio (HR) to select the top #Nf features for model building  
## Extract hazard ratio
HR <-
  as.numeric(levels(rfiltered_uv_results$HR))[rfiltered_uv_results$HR]
## Create modified HR for ranking
inverseHR <- function(x) {
  if (x < 1) {
    return(1 / x)
  } else{
    return(x)
  }
}
IHR <- sapply(HR, inverseHR)
## sort results by modified HR
rsorted_uv_results <-
  sort(IHR, index.return = TRUE, decreasing = TRUE)

featuresP <-
  names(rfiltered_uv_results$beta[rsorted_uv_results$ix])
# featuresP <- featuresP[-4]

TDTC <- cbind(TDT,DT[featuresP])
fmlac <-
  as.formula(paste("TSurvObjT ~ ", paste((
    names(TDTC)
  ), collapse = "+")))

coxRegNewTC <- cph(fmlac, data = TDTC, x = TRUE, y = TRUE)
predictionsTC <- survest(coxRegNewTC, TDTC, times = 24, se.fit = TRUE)
medianPSPC <- median(predictionsTC$surv)
ModelStrataTC <-
  as.vector(as.numeric(predictionsTC$surv > medianPSPC))
CItC <- rcorr.cens(predictionsTC$surv, TSurvObjT)["C Index"]
sdfTC <- survdiff(TSurvObjT ~ ModelStrataTC)
PvalTC <- 1 - pchisq(sdfTC$chisq, length(sdfTC$n) - 1)


DV <- ProspClinData[,c("clinical_Age_at_Diagnosis","clinical_Sex","patho_HPV_Status_Final")]
DV$Stage <- as.numeric(ProspClinData$ctn_Stage_at_Diagnosis_8Edition) < 5
DV$SmokeCur <- ProspClinData$risk_Smoker_at_Time_of_Diagnosis =='Current'
DV$SmokeCurFor <- ProspClinData$risk_Smoker_at_Time_of_Diagnosis == 'Current'|ProspClinData$risk_Smoker_at_Time_of_Diagnosis == 'Former'
DV$RegionOR <- ProspClinData$ctn_Tumor_Region == 'Oropharynx'
DV$RegionOC <- ProspClinData$ctn_Tumor_Region == 'Oral Cavity'
DV$RegionHY <- ProspClinData$ctn_Tumor_Region == 'Hypopharynx'
DV$RegionLA <- ProspClinData$ctn_Tumor_Region == 'Larynx'

TDVC <- cbind(TDV,DV[featuresP])
predictionsVC <- survest(coxRegNewTC, TDVC, times = 24, se.fit = TRUE)
ScoreVC <- predictionsV$surv
ModelStrataVC <-
  as.vector(as.numeric(predictionsVC$surv > medianPSPC))
sdfVC <- survdiff(TSurvObjV ~ ModelStrataVC)
PvalVC <- 1 - pchisq(sdfVC$chisq, length(sdfVC$n) - 1)

GenData <- read.csv("R:/Simon/BD2DECIDE/genomic_dataset_october.csv",header = TRUE, sep = ";", quote = "\"", dec = ".", fill = TRUE, comment.char = "")
GenRetro1 <- merge(
  RetroClinData,
  GenData,
  by.x = "Patient_ID",  #columnname of patientID in 'SurvivalData' used for merging
  by.y = "Patient_ID",   #columnname of patientID in 'Radiomicsdata' used for merging
  all = FALSE # use only matching rows
)
GenRetro <- GenRetro1[,14:38]

GenProsp1 <- merge(
  ProspClinData,
  GenData,
  by.x = "Patient_ID",  #columnname of patientID in 'SurvivalData' used for merging
  by.y = "Patient_ID",   #columnname of patientID in 'Radiomicsdata' used for merging
  all = FALSE # use only matching rows
)
GenProsp<- GenProsp1[,14:38]

TDTG <- TDTC
TDTG$INT_172_HNC_model <- as.numeric(as.character(GenRetro$INT_172_HNC_model))
# TDTG$INT_Clusters_OP_HPVpos <- as.numeric(GenRetro$INT_Clusters_OP_HPVpos)
GenRetro1 <- GenRetro1[-which(is.na (TDTG$INT_172_HNC_model)),]
# GenRetro1 <- GenRetro1[-which(TDTG$INT_Clusters_OP_HPVpos==1),]
TSurvObjTG <- TSurvObjT[-which(is.na (TDTG$INT_172_HNC_model)),]
# TSurvObjTG <- TSurvObjT[-which(TDTG$INT_Clusters_OP_HPVpos==1),]
TDTG <- TDTG[-which(is.na (TDTG$INT_172_HNC_model)),]
# TDTG <- TDTG[-which(TDTG$INT_Clusters_OP_HPVpos==1),]

fmlag <-
  as.formula(paste("TSurvObjTG ~ ", paste((
    c(names(TDTG),"INT_172_HNC_model")
  ), collapse = "+")))

coxRegNewTG <- cph(fmlag, data = TDTG, x = TRUE, y = TRUE)
predictionsTG <- survest(coxRegNewTG, TDTG, times = 24, se.fit = TRUE)
medianPSPG <- median(predictionsTG$surv)
ModelStrataTG <-
  as.vector(as.numeric(predictionsTG$surv > medianPSPG))
CItG <- rcorr.cens(predictionsTG$surv, TSurvObjTG)["C Index"]
sdfTG <- survdiff(TSurvObjTG ~ ModelStrataTG)
PvalTG <- 1 - pchisq(sdfTG$chisq, length(sdfTG$n) - 1)

TDVG <- TDVC
TDVG$INT_172_HNC_model  <- as.numeric(as.character(GenProsp$INT_172_HNC_model))
# TDVG$INT_172_HNC_model <- as.numeric(GenProsp$INT_172_HNC_model)
GenProsp1 <- GenProsp1[-which(is.na (TDVG$INT_172_HNC_model)),]
# GenProsp1 <- GenProsp1[-which(TDVG$INT_172_HNC_model==1),]
TSurvObjVG <- TSurvObjV[-which(is.na (TDVG$INT_172_HNC_model)),]
# TSurvObjVG <- TSurvObjV[-which(TDVG$INT_172_HNC_model==1),]
TDVG <- TDVG[-which(is.na (TDVG$INT_172_HNC_model)),]
# TDVG <- TDVG[-which(TDVG$INT_172_HNC_model==1),]

predictionsVG <- survest(coxRegNewTG, TDVG, times = 24, se.fit = TRUE)
ScoreVG <- predictionsVG$surv
ModelStrataVG <-
  as.vector(as.numeric(predictionsVG$surv > medianPSPG))
CIvG <- rcorr.cens(predictionsVG$surv, TSurvObjVG)["C Index"]
sdfVG <- survdiff(TSurvObjVG ~ ModelStrataVG)
PvalVG <- 1 - pchisq(sdfVG$chisq, length(sdfVG$n) - 1)

RetroResults <- cbind(as.character(RetroDataCom$Patient_ID),predictionsT$surv,ModelStrataT,predictionsTC$surv,ModelStrataTC)
ProspResults <- cbind(as.character(ProspData$Patient_ID),predictionsV$surv,ModelStrataV,predictionsVC$surv,ModelStrataVC)
TotalResults <- rbind(RetroResults,ProspResults)
write.csv(TotalResults, file = "R:/Simon/BD2DECIDE/Workspaces/OCResults.csv")

RetroGResults <- cbind(as.character(GenRetro1$Patient_ID),predictionsTG$surv,ModelStrataTG)
ProspGResults <- cbind(as.character(GenProsp1$Patient_ID),predictionsVG$surv,ModelStrataVG)
TotalGResults <- rbind(RetroGResults,ProspGResults)
write.csv(TotalGResults, file = "R:/Simon/BD2DECIDE/Workspaces/INT_172_HNC_model.csv")

CIv <- rcorr.cens(predictionsV$surv, TSurvObjV)["C Index"]

CIvC <- rcorr.cens(predictionsVC$surv, TSurvObjV)["C Index"]

FeatureListO
CIv
PvalV
CIe
PvalE
featuresP
CIvC
PvalVC
names(TDTG)
CIvG
PvalVG

coxRegNewT
coxRegNewTC
coxRegNewTG

TDE <- DesignRadDataS[FeatureListO]
predictionsE <- survest(coxRegNewT, TDE, times = 24, se.fit = TRUE)
ScoreE <- predictionsE$surv
ModelStrataE <-
  as.vector(as.numeric(predictionsE$surv > medianPSP))
TSurvObjE <- Surv(ExtClin$Survival_Time/30.44,ExtClin$Survival_Status)
sdfE <- survdiff(TSurvObjE ~ ModelStrataE)
PvalE <- 1 - pchisq(sdfE$chisq, length(sdfE$n) - 1)
CIe <- rcorr.cens(predictionsE$surv, TSurvObjE)["C Index"]

KaplanMeierCurveRT<-survfit(TSurvObjT ~ ModelStrataT, data = TDT)
KaplanMeierCurveRV<-survfit(TSurvObjV ~ ModelStrataV, data = TDV)
KaplanMeierCurveRE<-survfit(TSurvObjE ~ ModelStrataE, data = TDE)

ggsurvRadT<-ggsurvplot(
  KaplanMeierCurveRT,        # survfit object with calculated statistics.
  pval = TRUE,              # show p-value of log-rank test.
  xlim = c(0, 48),          # set x-axis limits
  conf.int = TRUE,          # show confidence intervals for
  # point estimates of survival curves.
  conf.int.style = "step",  # customize style of confidence intervals
  xlab = "Time in months",  # customize X axis label.
  break.time.by = 12,       # break X axis in time intervals.
  ggtheme = theme_bw(),     # customize plot and risk table with a theme.
  risk.table = "abs_pct",   # absolute number and percentage at risk.
  risk.table.y.text.col = T,# colour risk table text annotations.
  risk.table.y.text = FALSE,# show bars instead of names in text annotations
  # in legend of risk table.
  ncensor.plot = FALSE,      # plot the number of censored subjects at time t
  surv.median.line = "hv",  # add the median survival pointer.
  legend.labs =
    c("High-risk", "Low-risk"),    # change legend labels.
  palette =
    c("#E7B800", "#2E9FDF") # custom color palettes.
)


ggsurvRadT$plot <- ggsurvRadT$plot+ 
  ggplot2::annotate("text", 
                    x = 3.7, y = 0.12, # x and y coordinates of the text
                    label = paste("CI =", toString(round(CIt, digits = 3))), size = 5)
ggsurvRadT

ggsurvRad<-ggsurvplot(
  KaplanMeierCurveRV,        # survfit object with calculated statistics.
  pval = TRUE,              # show p-value of log-rank test.
  xlim = c(0, 48),          # set x-axis limits
  conf.int = TRUE,          # show confidence intervals for
  # point estimates of survival curves.
  conf.int.style = "step",  # customize style of confidence intervals
  xlab = "Time in months",  # customize X axis label.
  break.time.by = 12,       # break X axis in time intervals.
  ggtheme = theme_bw(),     # customize plot and risk table with a theme.
  risk.table = "abs_pct",   # absolute number and percentage at risk.
  risk.table.y.text.col = T,# colour risk table text annotations.
  risk.table.y.text = FALSE,# show bars instead of names in text annotations
  # in legend of risk table.
  ncensor.plot = FALSE,      # plot the number of censored subjects at time t
  surv.median.line = "hv",  # add the median survival pointer.
  legend.labs =
    c("High-risk", "Low-risk"),    # change legend labels.
  palette =
    c("#E7B800", "#2E9FDF") # custom color palettes.
)


ggsurvRad$plot <- ggsurvRad$plot+ 
  ggplot2::annotate("text", 
                    x = 3.7, y = 0.12, # x and y coordinates of the text
                    label = paste("CI =", toString(round(CIv, digits = 3))), size = 5)
ggsurvRad

ggsurvRad<-ggsurvplot(
  KaplanMeierCurveRE,        # survfit object with calculated statistics.
  pval = TRUE,              # show p-value of log-rank test.
  xlim = c(0, 48),          # set x-axis limits
  conf.int = TRUE,          # show confidence intervals for
  # point estimates of survival curves.
  conf.int.style = "step",  # customize style of confidence intervals
  xlab = "Time in months",  # customize X axis label.
  break.time.by = 12,       # break X axis in time intervals.
  ggtheme = theme_bw(),     # customize plot and risk table with a theme.
  risk.table = "abs_pct",   # absolute number and percentage at risk.
  risk.table.y.text.col = T,# colour risk table text annotations.
  risk.table.y.text = FALSE,# show bars instead of names in text annotations
  # in legend of risk table.
  ncensor.plot = FALSE,      # plot the number of censored subjects at time t
  surv.median.line = "hv",  # add the median survival pointer.
  legend.labs =
    c("High-risk", "Low-risk"),    # change legend labels.
  palette =
    c("#E7B800", "#2E9FDF") # custom color palettes.
)


ggsurvRad$plot <- ggsurvRad$plot+ 
  ggplot2::annotate("text", 
                    x = 3.7, y = 0.12, # x and y coordinates of the text
                    label = paste("CI =", toString(round(CIe, digits = 3))), size = 5)
ggsurvRad

KaplanMeierCurveRT<-survfit(TSurvObjT ~ ModelStrataTC, data = TDTC)
KaplanMeierCurveRV<-survfit(TSurvObjV ~ ModelStrataVC, data = TDVC)

ggsurvRadT<-ggsurvplot(
  KaplanMeierCurveRT,        # survfit object with calculated statistics.
  pval = TRUE,              # show p-value of log-rank test.
  xlim = c(0, 48),          # set x-axis limits
  conf.int = TRUE,          # show confidence intervals for
  # point estimates of survival curves.
  conf.int.style = "step",  # customize style of confidence intervals
  xlab = "Time in months",  # customize X axis label.
  break.time.by = 12,       # break X axis in time intervals.
  ggtheme = theme_bw(),     # customize plot and risk table with a theme.
  risk.table = "abs_pct",   # absolute number and percentage at risk.
  risk.table.y.text.col = T,# colour risk table text annotations.
  risk.table.y.text = FALSE,# show bars instead of names in text annotations
  # in legend of risk table.
  ncensor.plot = FALSE,      # plot the number of censored subjects at time t
  surv.median.line = "hv",  # add the median survival pointer.
  legend.labs =
    c("High-risk", "Low-risk"),    # change legend labels.
  palette =
    c("#E7B800", "#2E9FDF") # custom color palettes.
)


ggsurvRadT$plot <- ggsurvRadT$plot+ 
  ggplot2::annotate("text", 
                    x = 3.7, y = 0.12, # x and y coordinates of the text
                    label = paste("CI =", toString(round(CItC, digits = 3))), size = 5)
ggsurvRadT

ggsurvRad<-ggsurvplot(
  KaplanMeierCurveRV,        # survfit object with calculated statistics.
  pval = TRUE,              # show p-value of log-rank test.
  xlim = c(0, 48),          # set x-axis limits
  conf.int = TRUE,          # show confidence intervals for
  # point estimates of survival curves.
  conf.int.style = "step",  # customize style of confidence intervals
  xlab = "Time in months",  # customize X axis label.
  break.time.by = 12,       # break X axis in time intervals.
  ggtheme = theme_bw(),     # customize plot and risk table with a theme.
  risk.table = "abs_pct",   # absolute number and percentage at risk.
  risk.table.y.text.col = T,# colour risk table text annotations.
  risk.table.y.text = FALSE,# show bars instead of names in text annotations
  # in legend of risk table.
  ncensor.plot = FALSE,      # plot the number of censored subjects at time t
  surv.median.line = "hv",  # add the median survival pointer.
  legend.labs =
    c("High-risk", "Low-risk"),    # change legend labels.
  palette =
    c("#E7B800", "#2E9FDF") # custom color palettes.
)


ggsurvRad$plot <- ggsurvRad$plot+ 
  ggplot2::annotate("text", 
                    x = 3.7, y = 0.12, # x and y coordinates of the text
                    label = paste("CI =", toString(round(CIvC, digits = 3))), size = 5)
ggsurvRad

KaplanMeierCurveRT<-survfit(TSurvObjTG ~ ModelStrataTG, data = TDTG)
KaplanMeierCurveRV<-survfit(TSurvObjVG ~ ModelStrataVG, data = TDVG)

ggsurvRadT<-ggsurvplot(
  KaplanMeierCurveRT,        # survfit object with calculated statistics.
  pval = TRUE,              # show p-value of log-rank test.
  xlim = c(0, 48),          # set x-axis limits
  conf.int = TRUE,          # show confidence intervals for
  # point estimates of survival curves.
  conf.int.style = "step",  # customize style of confidence intervals
  xlab = "Time in months",  # customize X axis label.
  break.time.by = 12,       # break X axis in time intervals.
  ggtheme = theme_bw(),     # customize plot and risk table with a theme.
  risk.table = "abs_pct",   # absolute number and percentage at risk.
  risk.table.y.text.col = T,# colour risk table text annotations.
  risk.table.y.text = FALSE,# show bars instead of names in text annotations
  # in legend of risk table.
  ncensor.plot = FALSE,      # plot the number of censored subjects at time t
  surv.median.line = "hv",  # add the median survival pointer.
  legend.labs =
    c("High-risk", "Low-risk"),    # change legend labels.
  palette =
    c("#E7B800", "#2E9FDF") # custom color palettes.
)


ggsurvRadT$plot <- ggsurvRadT$plot+ 
  ggplot2::annotate("text", 
                    x = 3.7, y = 0.12, # x and y coordinates of the text
                    label = paste("CI =", toString(round(CItG, digits = 3))), size = 5)
ggsurvRadT

ggsurvRad<-ggsurvplot(
  KaplanMeierCurveRV,        # survfit object with calculated statistics.
  pval = TRUE,              # show p-value of log-rank test.
  xlim = c(0, 48),          # set x-axis limits
  conf.int = TRUE,          # show confidence intervals for
  # point estimates of survival curves.
  conf.int.style = "step",  # customize style of confidence intervals
  xlab = "Time in months",  # customize X axis label.
  break.time.by = 12,       # break X axis in time intervals.
  ggtheme = theme_bw(),     # customize plot and risk table with a theme.
  risk.table = "abs_pct",   # absolute number and percentage at risk.
  risk.table.y.text.col = T,# colour risk table text annotations.
  risk.table.y.text = FALSE,# show bars instead of names in text annotations
  # in legend of risk table.
  ncensor.plot = FALSE,      # plot the number of censored subjects at time t
  surv.median.line = "hv",  # add the median survival pointer.
  legend.labs =
    c("High-risk", "Low-risk"),    # change legend labels.
  palette =
    c("#E7B800", "#2E9FDF") # custom color palettes.
)


ggsurvRad$plot <- ggsurvRad$plot+ 
  ggplot2::annotate("text", 
                    x = 3.7, y = 0.12, # x and y coordinates of the text
                    label = paste("CI =", toString(round(CIvG, digits = 3))), size = 5)
ggsurvRad

ModelStrata7 <- as.numeric(ProspClinData$ctn_Stage_at_Diagnosis_7Edition) < 2 

CI7 <- rcorr.cens(ModelStrata7, TSurvObjV)["C Index"]
sdfTC <- survdiff(TSurvObjT ~ ModelStrataTC)
PvalTC <- 1 - pchisq(sdfTC$chisq, length(sdfTC$n) - 1)

TD8 <- ProspData
TD8$ModelStrata7 <- ModelStrata7

ModelStrata8 <- as.numeric(ProspData$ctn_Stage_at_Diagnosis_8Edition) < 5 

CI8 <- rcorr.cens(ModelStrata8, TSurvObjV)["C Index"]
sdfTC <- survdiff(TSurvObjT ~ ModelStrataTC)
PvalTC <- 1 - pchisq(sdfTC$chisq, length(sdfTC$n) - 1)

TD8$ModelStrata8 <- ModelStrata8

KaplanMeierCurve7<-survfit(TSurvObjV ~ ModelStrata7, data = TD8)
KaplanMeierCurve8<-survfit(TSurvObjV ~ ModelStrata8, data = TD8)

ggsurv7<-ggsurvplot(
  KaplanMeierCurve7,        # survfit object with calculated statistics.
  pval = TRUE,              # show p-value of log-rank test.
  xlim = c(0, 96),          # set x-axis limits
  conf.int = TRUE,          # show confidence intervals for
  # point estimates of survival curves.
  conf.int.style = "step",  # customize style of confidence intervals
  xlab = "Time in months",  # customize X axis label.
  break.time.by = 12,       # break X axis in time intervals.
  ggtheme = theme_bw(),     # customize plot and risk table with a theme.
  risk.table = "abs_pct",   # absolute number and percentage at risk.
  risk.table.y.text.col = T,# colour risk table text annotations.
  risk.table.y.text = FALSE,# show bars instead of names in text annotations
  # in legend of risk table.
  ncensor.plot = FALSE,      # plot the number of censored subjects at time t
  surv.median.line = "hv",  # add the median survival pointer.
  legend.labs =
    c("Group1", "Group2"),    # change legend labels.
  palette =
    c("#E7B800", "#2E9FDF") # custom color palettes.
)


ggsurv7$plot <- ggsurv7$plot+ 
  ggplot2::annotate("text", 
                    x = 3.7, y = 0.12, # x and y coordinates of the text
                    label = paste("CI =", toString(round(CI7, digits = 3))), size = 5)
ggsurv7

ggsurv8<-ggsurvplot(
  KaplanMeierCurve8,        # survfit object with calculated statistics.
  pval = TRUE,              # show p-value of log-rank test.
  xlim = c(0, 48),          # set x-axis limits
  conf.int = TRUE,          # show confidence intervals for
  # point estimates of survival curves.
  conf.int.style = "step",  # customize style of confidence intervals
  xlab = "Time in months",  # customize X axis label.
  break.time.by = 12,       # break X axis in time intervals.
  ggtheme = theme_bw(),     # customize plot and risk table with a theme.
  risk.table = "abs_pct",   # absolute number and percentage at risk.
  risk.table.y.text.col = T,# colour risk table text annotations.
  risk.table.y.text = FALSE,# show bars instead of names in text annotations
  # in legend of risk table.
  ncensor.plot = FALSE,      # plot the number of censored subjects at time t
  surv.median.line = "hv",  # add the median survival pointer.
  legend.labs =
    c("Stage IV", "Stage III"),    # change legend labels.
  palette =
    c("#E7B800", "#2E9FDF") # custom color palettes.
)


ggsurv8$plot <- ggsurv8$plot+ 
  ggplot2::annotate("text", 
                    x = 3.6, y = 0.12, # x and y coordinates of the text
                    label = paste("CI =", toString(round(CI8, digits = 3))), size = 5)
ggsurv8

TDT$Shape_volume <- RetroRadDataS$Shape_volume
fmlavol <- as.formula(paste("TSurvObjT ~ Shape_volume"))

ddist <- datadist(TDT)
options(datadist='ddist')
coxRegNewvol <- cph(fmlavol, data = TDT, x = TRUE, y = TRUE)
predictionsvol <- survest(coxRegNewvol, TDT, times = 24, se.fit = TRUE)
medianPSPvol <- median(predictionsvol$surv)
ModelStratavol <-
  as.vector(as.numeric(predictionsvol$surv > medianPSPvol))
CITvol <- rcorr.cens(predictionsvol$surv, TSurvObjT)["C Index"]

TDV$Shape_volume <- ProspRadDataS$Shape_volume
predictionsVvol <- survest(coxRegNewvol, TDV, times = 24, se.fit = TRUE)
ScoreVvol <- predictionsVvol$surv
ModelStrataVvol <-
  as.vector(as.numeric(predictionsVvol$surv > medianPSPvol))
CIVvol <- rcorr.cens(predictionsVvol$surv, TSurvObjV)["C Index"]

KaplanMeierCurveRTVol<-survfit(TSurvObjT ~ ModelStratavol, data = TDT)

ggsurvTvol<-ggsurvplot(
  KaplanMeierCurveRTVol,        # survfit object with calculated statistics.
  pval = TRUE,              # show p-value of log-rank test.
  xlim = c(0, 48),          # set x-axis limits
  conf.int = TRUE,          # show confidence intervals for
  # point estimates of survival curves.
  conf.int.style = "step",  # customize style of confidence intervals
  xlab = "Time in months",  # customize X axis label.
  break.time.by = 12,       # break X axis in time intervals.
  ggtheme = theme_bw(),     # customize plot and risk table with a theme.
  risk.table = "abs_pct",   # absolute number and percentage at risk.
  risk.table.y.text.col = T,# colour risk table text annotations.
  risk.table.y.text = FALSE,# show bars instead of names in text annotations
  # in legend of risk table.
  ncensor.plot = FALSE,      # plot the number of censored subjects at time t
  surv.median.line = "hv",  # add the median survival pointer.
  legend.labs =
    c("Group1", "Group2"),    # change legend labels.
  palette =
    c("#E7B800", "#2E9FDF") # custom color palettes.
)


ggsurvTvol$plot <- ggsurvTvol$plot+ 
  ggplot2::annotate("text", 
                    x = 3.7, y = 0.12, # x and y coordinates of the text
                    label = paste("CI =", toString(round(CITvol, digits = 3))), size = 5)
ggsurvTvol

KaplanMeierCurveRVol<-survfit(TSurvObjV ~ ModelStrataVvol, data = TDV)

ggsurvvol<-ggsurvplot(
  KaplanMeierCurveRVol,        # survfit object with calculated statistics.
  pval = TRUE,              # show p-value of log-rank test.
  xlim = c(0, 48),          # set x-axis limits
  conf.int = TRUE,          # show confidence intervals for
  # point estimates of survival curves.
  conf.int.style = "step",  # customize style of confidence intervals
  xlab = "Time in months",  # customize X axis label.
  break.time.by = 12,       # break X axis in time intervals.
  ggtheme = theme_bw(),     # customize plot and risk table with a theme.
  risk.table = "abs_pct",   # absolute number and percentage at risk.
  risk.table.y.text.col = T,# colour risk table text annotations.
  risk.table.y.text = FALSE,# show bars instead of names in text annotations
  # in legend of risk table.
  ncensor.plot = FALSE,      # plot the number of censored subjects at time t
  surv.median.line = "hv",  # add the median survival pointer.
  legend.labs =
    c("High-risk", "Low-risk"),    # change legend labels.
  palette =
    c("#E7B800", "#2E9FDF") # custom color palettes.
)


ggsurvvol$plot <- ggsurvvol$plot+ 
  ggplot2::annotate("text", 
                    x = 4.2, y = 0.12, # x and y coordinates of the text
                    label = paste("CI =", toString(round(CIVvol, digits = 3))), size = 5)
ggsurvvol

FeatureListStage <- c(FeatureListO,'Stage_8')

fmla <-
  as.formula(paste("TSurvObjT ~ ", paste((
    FeatureListStage
  ), collapse = "+")))

TDT$Stage_8 <-  as.numeric(RetroClinData$ctn_Stage_at_Diagnosis_8Edition) < 5 
coxRegNewT <- cph(fmla, data = TDT, x = TRUE, y = TRUE)
predictionsT <- survest(coxRegNewT, TDT, times = 24, se.fit = TRUE)
medianPSP <- median(predictionsT$surv)
ModelStrataT <-
  as.vector(as.numeric(predictionsT$surv > medianPSP))
CIt <- rcorr.cens(predictionsT$surv, TSurvObjT)["C Index"]
sdfT <- survdiff(TSurvObjT ~ ModelStrataT)
PvalT <- 1 - pchisq(sdfT$chisq, length(sdfT$n) - 1)

TDV <- ProspRadDataS[FeatureListO]
TDV$Stage_8 <-  as.numeric(ProspClinData$ctn_Stage_at_Diagnosis_8Edition) < 5 
predictionsV <- survest(coxRegNewT, TDV, times = 24, se.fit = TRUE)
ScoreV <- predictionsV$surv
ModelStrataV <-
  as.vector(as.numeric(predictionsV$surv > medianPSP))

CIv <- rcorr.cens(predictionsV$surv, TSurvObjV)["C Index"]
KaplanMeierCurveRT<-survfit(TSurvObjT ~ ModelStrataT, data = TDT)
KaplanMeierCurveRV<-survfit(TSurvObjV ~ ModelStrataV, data = TDV)

ggsurvRadT<-ggsurvplot(
  KaplanMeierCurveRT,        # survfit object with calculated statistics.
  pval = TRUE,              # show p-value of log-rank test.
  xlim = c(0, 96),          # set x-axis limits
  conf.int = TRUE,          # show confidence intervals for
  # point estimates of survival curves.
  conf.int.style = "step",  # customize style of confidence intervals
  xlab = "Time in months",  # customize X axis label.
  break.time.by = 12,       # break X axis in time intervals.
  ggtheme = theme_bw(),     # customize plot and risk table with a theme.
  risk.table = "abs_pct",   # absolute number and percentage at risk.
  risk.table.y.text.col = T,# colour risk table text annotations.
  risk.table.y.text = FALSE,# show bars instead of names in text annotations
  # in legend of risk table.
  ncensor.plot = FALSE,      # plot the number of censored subjects at time t
  surv.median.line = "hv",  # add the median survival pointer.
  legend.labs =
    c("Group1", "Group2"),    # change legend labels.
  palette =
    c("#E7B800", "#2E9FDF") # custom color palettes.
)


ggsurvRadT$plot <- ggsurvRadT$plot+ 
  ggplot2::annotate("text", 
                    x = 3.7, y = 0.12, # x and y coordinates of the text
                    label = paste("CI =", toString(round(CIt, digits = 3))), size = 5)
ggsurvRadT

ggsurvRad<-ggsurvplot(
  KaplanMeierCurveRV,        # survfit object with calculated statistics.
  pval = TRUE,              # show p-value of log-rank test.
  xlim = c(0, 48),          # set x-axis limits
  conf.int = TRUE,          # show confidence intervals for
  # point estimates of survival curves.
  conf.int.style = "step",  # customize style of confidence intervals
  xlab = "Time in months",  # customize X axis label.
  break.time.by = 12,       # break X axis in time intervals.
  ggtheme = theme_bw(),     # customize plot and risk table with a theme.
  risk.table = "abs_pct",   # absolute number and percentage at risk.
  risk.table.y.text.col = T,# colour risk table text annotations.
  risk.table.y.text = FALSE,# show bars instead of names in text annotations
  # in legend of risk table.
  ncensor.plot = FALSE,      # plot the number of censored subjects at time t
  surv.median.line = "hv",  # add the median survival pointer.
  legend.labs =
    c("High-risk", "Low-risk"),    # change legend labels.
  palette =
    c("#E7B800", "#2E9FDF") # custom color palettes.
)


ggsurvRad$plot <- ggsurvRad$plot+ 
  ggplot2::annotate("text", 
                    x = 3.7, y = 0.12, # x and y coordinates of the text
                    label = paste("CI =", toString(round(CIv, digits = 3))), size = 5)
ggsurvRad


