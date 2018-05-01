#source('~/Dropbox/APPM/APPM4580/FinalProject/Project.R')
rm(list = ls())
library(caret)
library(dummies)
set.seed(42)

##################################################
#                 Load the Data         
##################################################
# Columns: Date, Block, IUCR, Primary.Type, Description, Location.Description, 
#			Arrest, Domestic, Beat, District, Community.Area, Year, Latitude, Longitude
dat = read.csv(file='/Users/mariaknigge/Dropbox/APPM/APPM4580/FinalProject/Chicago_Crime.csv')
n = length(dat[,1])

##################################################
#                 Preprocessing         
##################################################

# Downsizing: Create random sample of data
ds.p = 0.0056
downsample = sample(c(TRUE, FALSE), n, replace = TRUE, prob = c(ds.p, (1 - ds.p)))
dat = droplevels(dat[downsample,])
n = length(dat[,1])

# Translate Variables / Create New
dat$Time = as.factor(format(as.POSIXct(strftime(dat$Date, '%Y-%m-%d %H:%M:%S')), format = '%H:%M'))
dat$Hour = as.factor(format(as.POSIXct(strftime(dat$Date, '%Y-%m-%d %H:%M:%S')), format = '%H'))
dat$Day = as.factor(format(as.POSIXlt(strftime(dat$Date, '%Y-%m-%d %H:%M:%S')), format = '%d'))
dat$Date = as.Date(dat$Date)
dat$Month = as.factor(months(dat$Date, T))
dat$Year = as.factor(dat$Year)
dat$Arrest = as.factor(as.integer(as.logical(dat$Arrest)))
dat$Domestic = as.factor(as.integer(as.logical(dat$Domestic)))
dat$Street = as.factor(sapply(dat$Block, function(i) paste(strsplit(as.character(dat$Block[i]), ' ')[[1]][-1], collapse = ' ')))

# Exclude redundant columns: Primary.Type, Description, FBI.Code, Month, Block
usecols = c('Month', 'Day', 'Year', 'Hour', 'IUCR', 'Arrest', 'Domestic', 'Latitude', 'Longitude')
#usecols = c('Date', 'MonthDay', 'Year', 'Hour', 'Time', 'Month', 'IUCR', 'Arrest', 'Domestic', 'Block', 'Street', 'Location.Description', 'Beat', 'District', 'Community.Area', 'Latitude', 'Longitude')
dat = dat[order(dat$Date), usecols]

##################################################
#                   Training
##################################################
#..................Clustering.....................
# Reduce number of latitude and longitude coordinates


#.....................PCA.........................
pca.dat = dummy.data.frame(dat[, usecols[-5]])

# Split into train/test data sets
training = sample(c(TRUE, FALSE), n, replace = TRUE, prob = c(0.9, 0.1))
train = dat[training, usecols]
test = dat[!training, usecols]
#train = pca.dat[training,]
#test = pca.dat[!training,]

#prin_comp = prcomp(pca.train, scale. = T)

#.....................KNN.........................
train.ctrl = trainControl(method = 'repeatedcv', number = 5, repeats = 3)
set.seed(314159)
#knn.fit = train(Arrest~., data = train, method = 'knn', trControl = train.ctrl, preProcess = c('center', 'scale'), tuneLength = 5)

#.....................SVM.........................
# Split into train/test data sets
#training = sample(c(TRUE, FALSE), n, replace = TRUE, prob = c(0.9, 0.1))
#train = dat[training,]
#test = dat[!training,]

grid.lin = expand.grid(C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 5))
#svm.Linear = train(Arrest~., data = train, method = 'svmLinear', trControl = train.ctrl, preProcess = c('center', 'scale'), tuneGrid = grid.lin, tuneLength = 5)
#test.pred.lin = predict(svm.Linear, newdata = test)
#confusionMatrix(test.pred.lin, test$Arrest)

grid.rad = expand.grid(sigma = c(0,0.01, 0.025, 0.05, 0.06, 0.075, 0.1, 0.25, 0.5, 0.75, 0.9), C = c(1, 1.25, 1.5, 1.75, 2, 5))
#svm.Radial = train(Arrest~., data = train, method = 'svmRadial', trControl=train.ctrl, preProcess = c("center", "scale"), tuneGrid = grid.rad, tuneLength = 5)
#test.pred.rad = predict(svm.Radial, newdata = test)
#confusionMatrix(test.pred.rad, test$Arrest)

grid.poly = expand.grid(degree = c(1, 2, 3, 4, 5), scale = c(0.001, 0.01, 0.05), C = c(0, 0.01, 0.1, 0.25, 0.5, 1, 1.5, 2))
#svm.Poly = train(Arrest~., data = train, method = 'svmPoly', trControl=train.ctrl, preProcess = c("center", "scale"), tuneGrid = grid.poly, tuneLength = 5)
#test.pred.poly = predict(svm.Poly, newdata = test)
#confusionMatrix(test.pred.rad, test$Arrest)