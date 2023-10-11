# MIS510
GermanCredit.df <- read.csv("GermanCredit.csv", header= TRUE)
View(GermanCredit.df) # View data set in a new tab
dim(GermanCredit.df) # Display the # of observations and variables
summary(GermanCredit.df) # simple statistics of each variable
head(GermanCredit.df) # view first 6 rows 
GermanCredit.df[1:10,32] #Show the first 10 rows of the Response Variable


#Load rpart and rpart.plot
library(rpart)
library(rpart.plot)
library(caret)
#Create simple names for variables
Response <- GermanCredit.df$RESPONSE #Outcome variable
Duration <- GermanCredit.df$DURATION #Predictor Variable
Amount <- GermanCredit.df$AMOUNT #Predictor variable
Install_Rate <- GermanCredit.df$INSTALL_RATE
Age <- GermanCredit.df$AGE # Numerical Predictor Variable
NewCar <- GermanCredit.df$NEW_CAR #Binary
Used_Car <- GermanCredit.df$USED_CAR #Binary
Chk_Acct <- GermanCredit.df$CHK_ACCT #Categorical
History <- GermanCredit.df$HISTORY #Categorical
Used_Car <- GermanCredit.df$USED_CAR #Binary
Furniture <- GermanCredit.df$FURNITURE #Binary
Radio.TV <- GermanCredit.df$RADIO.TV #Binary
Education <- GermanCredit.df$EDUCATION  #Binary
Retraining <- GermanCredit.df$RETRAINING  #Binary
Sav_Acct<- GermanCredit.df$SAV_ACCT  #Categorical
Employment <- GermanCredit.df$EMPLOYMENT #Predictor Variable
Male_Div <- GermanCredit.df$MALE_DIV #Binary
Male_Single <- GermanCredit.df$MALE_SINGLE #Binary
Male_Mar_Wid <- GermanCredit.df$MALE_MAR_or_WID #Binary
Co_Applicant <- GermanCredit.df$CO.APPLICANT #Binary
Guarantor <- GermanCredit.df$GUARANTOR #Binary
Present_Resident <- GermanCredit.df$PRESENT_RESIDENT  #Categorical
Real_Estate <- GermanCredit.df$REAL_ESTATE #Binary
Other_Install <- GermanCredit.df$OTHER_INSTALL #Binary
Rent <- GermanCredit.df$RENT #Binary
Own_Res <- GermanCredit.df$OWN_RES #Binary
Num_Credits <- GermanCredit.df$NUM_CREDITS #Numerical
Job <- GermanCredit.df$JOB #Categorical
Num_Dependents <- GermanCredit.df$NUM_DEPENDENTS #Numerical
Foreign <- GermanCredit.df$FOREIGN #Binary

#Create Dummy Variables
#Convert Binary and Categorical Variables
GCdummy <- model.matrix(~ 0 + ., data = GermanCredit.df)
GCdummy <- as.data.frame(GCdummy) #Convert Dummy Variables into a Data Frame
t(t(names(GCdummy))) #Verify names of Dummy data frame
head(GCdummy) #Verify Conversion
GCdummy <- GCdummy[, -15:-17]

#Divide the data into training and validation partitions
#Partition Preparation
set.seed(1) #To reproduce the same output of simulation studies
#Create partition
train.index <- sample(c(1:dim(GCdummy)[1]), 
                             dim(GCdummy)[1]*0.6) 
train.df <- GCdummy[train.index, ] #Training partition
valid.df <- GCdummy[-train.index, ] #Validation partition

#Build Classification Tree Using Training data
GC.Tree <- rpart(RESPONSE ~ .,
                 data  = train.df, method = "class", minbucket = 20) #Model
#plot tree for training data
prp(GC.Tree) #Display Training Data Tree

#Build CART using Validation data
GC.Tree.Valid <- rpart(RESPONSE ~ .,
                       data  = valid.df, method = "class", minbucket = 20) #Model
#Plot for Validation data
prp(GC.Tree.Valid) #Display Validation data Tree

#Performance Evaluation of CART
#Confusion Matrix using table()
GC.Tree.Predict <- predict(GC.Tree, newdata = train.df, type = "class")
table(GC.Tree.Predict)
GC.Tree.Predict.Valid <- predict(GC.Tree.Valid, newdata= valid.df, type = "class")
table(GC.Tree.Predict.Valid)
#Accuracy Calculations
table(train.df$RESPONSE, GC.Tree.Predict)
(111+351)/(111+90+48+351) #Determine Accuracy of training tree
table(valid.df$RESPONSE, GC.Tree.Predict.Valid)
(46+281)/(46+53+20+281) #Determine Accuracy of Validation Tree


#Neural Network
#Install packages
install.packages("jsonlite", type = "source")
library(jsonlite)
install.packages("neuralnet")
#Load Libraries
library(neuralnet)
library(nnet)
library(caret)
#Selected variables from Tree
vars<-c("train.df$CHK_ACCT", "train.df$DURATION", "train.df$GUARANTOR",
        "train.df$HISTORY", "train.df$AMOUNT", "train.df$REAL_ESTATE")

#Partition data
set.seed(2)

#Create Dummy Variables
#Convert Binary and Categorical Variables
GCdummynn <- model.matrix(~ 0 + ., data = GermanCredit.df)
GCdummynn <- as.data.frame(GCdummynn) #Convert Dummy Variables into a Data Frame
t(t(names(GCdummynn))) #Verify names of Dummy data frame
head(GCdummynn) #Verify Conversion
GCdummynn <- GCdummynn[, -15:-17]


#Create partition
nntrain.index <- sample(c(1:dim(GCdummynn)[1]), 
                      dim(GCdummynn)[1]*0.6) 
NNtrain.df <- GCdummynn[nntrain.index, ] #Training partition
NNvalid.df <- GCdummynn[-nntrain.index, ] #Validation partition

nn = neuralnet(RESPONSE ~ CHK_ACCT + DURATION + SAV_ACCT + HISTORY + AMOUNT,
               data = NNtrain.df, hidden = 2, 
               err.fct = "ce", 
               linear.output = FALSE) #Training model
prediction(nn)
plot(nn) #View training Network

nnvalid = neuralnet(RESPONSE ~ CHK_ACCT + AMOUNT,
                    data = NNvalid.df, hidden = 2, 
                    err.fct = "ce", linear.output = FALSE) #Validation Model
plot(nnvalid) #View validation network

plot(prediction(nn))
#Confusion Matrix using table()
nn.Predict <- predict(nn, newdata = NNtrain.df, type = "class")
table(nn.Predict)
nn.Predict.Valid <- predict(nnvalid, newdata= NNvalid.df, type = "class")
table(nn.Predict.Valid)

#Accuracy Calculations
table(RESPONSE, nn.Predict)

table(NNvalid.df$RESPONSE, nn.Predict.Valid)

#Neural Network Performance Evaluation
nn$net.result
NNtrain.df$RESPONSE #training set 
nnvalid$net.result[[1]] #validation set net result
nn3 = ifelse(nn$net.result[[1]]>0.5, 1, 0)
nn3[1:6] #first 6 rows
misClasificationError = mean(NNtrain.df$RESPONSE !=nn3)
misClasificationError
nn4 = ifelse(nnvalid$net.result[[1]]>0.5, 1, 0)
nn4[1:6] #first 6 rows
misClasificationError.valid = mean(NNvalid.df$RESPONSE !=nn4)
misClasificationError.valid

plot(GermanCredit.df[.])
ggpairs(GermanCredit.df[,.])
