# Building-models-using-many-variables

# load KKD Cup 2009 dataset

d <- read.table("orange_small_train.data", header=T, sep="\t", na.strings=c("NA",""))

churn <- read.table("orange_small_train_churn.labels.txt", header=F, sep="\t") 	# Note: 3 

d$churn <- churn$V1 	# Note: 4 

appetency <- read.table("orange_small_train_appetency.labels.txt", header=F, sep="\t")

d$appetency <- appetency$V1 	# Note: 5 

upselling <- read.table("orange_small_train_upselling.labels.txt", header=F, sep="\t")

d$upselling <- upselling$V1 	# Note: 6 

# define dataset into categorical and numerical varibles
# devide dataset into train, test, and calibartion
set.seed(729375) 	# Note: 7 
d$rgroup <- runif(dim(d)[[1]])
dTrainAll <- subset(d,rgroup<=0.9)
dTest <- subset(d,rgroup>0.9) 	# Note: 8 
outcomes=c('churn','appetency','upselling')
vars <- setdiff(colnames(dTrainAll),
                c(outcomes,'rgroup'))
catVars <- vars[sapply(dTrainAll[,vars],class) %in%
                  c('factor','character')] 	# Note: 9 
numericVars <- vars[sapply(dTrainAll[,vars],class) %in%
                      c('numeric','integer')] 	# Note: 10 
rm(list=c('d','churn','appetency','upselling')) 	# Note: 11 
outcome <- 'churn' 	# Note: 12 
pos <- '1' 	# Note: 13 
useForCal <- rbinom(n=dim(dTrainAll)[[1]],size=1,prob=0.5)>0 	# Note: 14 
dCal <- subset(dTrainAll,useForCal)
dTrain <- subset(dTrainAll,!useForCal)

# function to build single-variable models for categorical variables 
mkPredC <- function(outCol,varCol,appCol) { 	# Note: 1 
  pPos <- sum(outCol==pos)/length(outCol) 	# Note: 2 
  naTab <- table(as.factor(outCol[is.na(varCol)]))
  pPosWna <- (naTab/sum(naTab))[as.character(pos)] 	# Note: 3 
  vTab <- table(as.factor(outCol),varCol)
  pPosWv <- (vTab[pos,]+1.0e-3*pPos)/(colSums(vTab)+1.0e-3) 	# Note: 4 
  pred <- pPosWv[appCol] 	# Note: 5 
  pred[is.na(appCol)] <- pPosWna 	# Note: 6 
  pred[is.na(pred)] <- pPos 	# Note: 7 
  pred 	# Note: 8 
}

for(v in catVars) {
  pi <- paste('pred',v,sep='')
  dTrain[,pi] <- mkPredC(dTrain[,outcome],dTrain[,v],dTrain[,v])
  dCal[,pi] <- mkPredC(dTrain[,outcome],dTrain[,v],dCal[,v])
  dTest[,pi] <- mkPredC(dTrain[,outcome],dTrain[,v],dTest[,v])
}

# function to build single-variable models for numerical variables 
mkPredN <- function(outCol,varCol,appCol) {
  nval <- length(unique(varCol[!is.na(varCol)]))
  if(nval<=1) {
    pPos <- sum(outCol==pos)/length(outCol)
    return(pPos+numeric(length(appCol)))
  }
  cuts <- unique(as.numeric(quantile(varCol,
                                     probs=seq(0, 1, 0.1),na.rm=T)))
  varC <- cut(varCol,cuts)
  appC <- cut(appCol,cuts)
  mkPredC(outCol,varC,appC)
}
for(v in numericVars) {
  pi <- paste('pred',v,sep='')
  dTrain[,pi] <- mkPredN(dTrain[,outcome],dTrain[,v],dTrain[,v])
  dTest[,pi] <- mkPredN(dTrain[,outcome],dTrain[,v],dTest[,v])
  dCal[,pi] <- mkPredN(dTrain[,outcome],dTrain[,v],dCal[,v])
  aucTrain <- calcAUC(dTrain[,pi],dTrain[,outcome])
  if(aucTrain>=0.55) {
    aucCal <- calcAUC(dCal[,pi],dCal[,outcome])
    print(sprintf("%s, trainAUC: %4.3f calibrationAUC: %4.3f",
                  pi,aucTrain,aucCal))
  }
}

# function for variables selection
logLikelyhood <- function(outCol,predCol) { 	# Note: 1 
  sum(ifelse(outCol==pos,log(predCol),log(1-predCol)))
}
selVars <- c()
minStep <- 5
baseRateCheck <- logLikelyhood(dCal[,outcome],
                               sum(dCal[,outcome]==pos)/length(dCal[,outcome]))
for(v in catVars) {  	# Note: 2 
  pi <- paste('pred',v,sep='')
  liCheck <- 2*((logLikelyhood(dCal[,outcome],dCal[,pi]) -
                   baseRateCheck))
  if(liCheck>minStep) {
    print(sprintf("%s, calibrationScore: %g",
                  pi,liCheck))
    selVars <- c(selVars, pi)
  }
}
for(v in numericVars) { 	# Note: 3 
  pi <- paste('pred',v,sep='')
  liCheck <- 2*((logLikelyhood(dCal[,outcome],dCal[,pi]) -
                   baseRateCheck))
  if(liCheck>=minStep) {
    print(sprintf("%s, calibrationScore: %g",
                  pi,liCheck))
    selVars <- c(selVars,pi)
  }
}

# function to calculate AUC value
library('ROCR')
calcAUC <- function(predcol,outcol) {
  perf <- performance(prediction(predcol,outcol==pos),'auc')
  as.numeric(perf@y.values)
}
for(v in catVars) {
  pi <- paste('pred',v,sep='')
  aucTrain <- calcAUC(dTrain[,pi],dTrain[,outcome])
  if(aucTrain>=0.8) {
    aucCal <- calcAUC(dCal[,pi],dCal[,outcome])
    print(sprintf("%s, trainAUC: %4.3f calibrationAUC: %4.3f",
                  pi,aucTrain,aucCal))
  }
}

################## Decision Tree #######################
# with all categorical and numerical variables
library('rpart')
fV <- paste(outcome,'>0 ~ ',
            paste(c(catVars,numericVars),collapse=' + '),sep='')
tmodel <- rpart(fV,data=dCal)
print(calcAUC(predict(tmodel,newdata=dTrain),dTrain[,outcome]))
print(calcAUC(predict(tmodel,newdata=dTest),dTest[,outcome]))
print(calcAUC(predict(tmodel,newdata=dCal),dCal[,outcome]))

# with selected categorical and numerical variables
f <- paste(outcome,'>0 ~ ',paste(selVars,collapse=' + '),sep='')
tmodel <- rpart(f,data=dCal,
                control=rpart.control(cp=0.001,minsplit=1000,
                                      minbucket=1000,maxdepth=5)
)
print(calcAUC(predict(tmodel,newdata=dTrain),dTrain[,outcome]))
print(calcAUC(predict(tmodel,newdata=dTest),dTest[,outcome]))
print(calcAUC(predict(tmodel,newdata=dCal),dCal[,outcome]))

print(tmodel)
par(cex=0.7)
plot(tmodel)
text(tmodel)


################# KNN ###########################
# KNN function
library('class')
nK <- 200
knnTrain <- dCal[,selVars]  	# Note: 1 
knnCl <- dCal[,outcome]==pos 	# Note: 2 
knnPred <- function(df) { 	# Note: 3 
  knnDecision <- knn(knnTrain,df,knnCl,k=nK,prob=T)
  ifelse(knnDecision==TRUE, 	# Note: 4 
         attributes(knnDecision)$prob,
         1-(attributes(knnDecision)$prob))
}
print(calcAUC(knnPred(dTrain[,selVars]),dTrain[,outcome]))
print(calcAUC(knnPred(dCal[,selVars]),dCal[,outcome]))
print(calcAUC(knnPred(dTest[,selVars]),dTest[,outcome]))

# platting 200-nearest neighbor performance 
library(ggplot2)
dCal$kpred <- knnPred(dCal[,selVars])
ggplot(data=dCal) +
  geom_density(aes(x=kpred,
                   color=as.factor(churn),linetype=as.factor(churn)))

# plotting the receiver operating characteristic curve 
plotROC <- function(predcol,outcol) {
  perf <- performance(prediction(predcol,outcol==pos),'tpr','fpr')
  pf <- data.frame(
    FalsePositiveRate=perf@x.values[[1]],
    TruePositiveRate=perf@y.values[[1]])
  ggplot() +
    geom_line(data=pf,aes(x=FalsePositiveRate,y=TruePositiveRate)) +
    geom_line(aes(x=c(0,1),y=c(0,1)))
}
print(plotROC(knnPred(dTest[,selVars]),dTest[,outcome]))

# plotting the performance of a logistic regression model 
gmodel <- glm(as.formula(f),data=dCal,family=binomial(link='logit'))
print(calcAUC(predict(gmodel,newdata=dTrain),dTrain[,outcome]))
print(calcAUC(predict(gmodel,newdata=dTest),dTest[,outcome]))
print(calcAUC(predict(gmodel,newdata=dCal),dCal[,outcome]))

################### Naive Bayes #########################
# Naive Bayes function
pPos <- sum(dCal[,outcome]==pos)/length(dCal[,outcome])
nBayes <- function(pPos,pf) { 	# Note: 1 
  pNeg <- 1 - pPos
  smoothingEpsilon <- 1.0e-5
  scorePos <- log(pPos + smoothingEpsilon) + 
    rowSums(log(pf/pPos + smoothingEpsilon)) 	# Note: 2 
  scoreNeg <- log(pNeg + smoothingEpsilon) +
    rowSums(log((1-pf)/(1-pPos) + smoothingEpsilon)) 	# Note: 3 
  m <- pmax(scorePos,scoreNeg)
  expScorePos <- exp(scorePos-m)
  expScoreNeg <- exp(scoreNeg-m) 	# Note: 4 
  expScorePos/(expScorePos+expScoreNeg) 	# Note: 5 
}
pVars <- paste('pred',c(numericVars,catVars),sep='')
dTrain$nbpredl <- nBayes(pPos,dTrain[,pVars])
dCal$nbpredl <- nBayes(pPos,dCal[,pVars])
dTest$nbpredl <- nBayes(pPos,dTest[,pVars]) 	# Note: 6 
print(calcAUC(dTrain$nbpredl,dTrain[,outcome]))
print(calcAUC(dCal$nbpredl,dCal[,outcome]))
print(calcAUC(dTest$nbpredl,dTest[,outcome]))

# using Naive Bayes packages
library('e1071')
lVars <- c(catVars,numericVars)
ff <- paste('as.factor(',outcome,'>0) ~ ',
            paste(lVars,collapse=' + '),sep='')
nbmodel <- naiveBayes(as.formula(ff),data=dCal)
dTrain$nbpred <- predict(nbmodel,newdata=dTrain,type='raw')[,'TRUE']
dCal$nbpred <- predict(nbmodel,newdata=dCal,type='raw')[,'TRUE']
dTest$nbpred <- predict(nbmodel,newdata=dTest,type='raw')[,'TRUE']
calcAUC(dTrain$nbpred,dTrain[,outcome])
calcAUC(dCal$nbpred,dCal[,outcome])
calcAUC(dTest$nbpred,dTest[,outcome])
