rm(list = ls())
graphics.off()

setwd("D:/school/statistical learning & data mining/final project")

library(readr)
BlackFriday <- read_csv("train.csv", 
                        col_types = cols(Product_Category_1 = col_character(), 
                                         Product_Category_2 = col_character(), 
                                         Product_Category_3 = col_character()))

#### preprocessing ####
b <- BlackFriday
for(i in 9:11){
  b = b[which(!(as.numeric(unlist(b[,i])) %in% c(19,20))),]
}

###################################################################
#### one-hot encode the categories ####
require(qdapTools)

product_cat.idx <- colnames(b) %in% c("Product_Category_1","Product_Category_2","Product_Category_3")
cat <- b[, product_cat.idx]
cat$join <- paste(cat$Product_Category_1, cat$Product_Category_2, cat$Product_Category_3, sep=",")

onehot <- mtabulate(strsplit(cat$join, ","))
onehot <- onehot[,as.character(sort(as.numeric(colnames(onehot[,!colnames(onehot) %in% c("NA")]))))]

colnames(onehot) <- paste("category_",colnames(onehot), sep="")
b <- cbind(b[,!product_cat.idx],onehot)

#### aggregate by user ID ####
data <- b[,c("User_ID","Gender","Age","Occupation","City_Category","Stay_In_Current_City_Years",
                       "Marital_Status","Purchase", paste("category_",c(1:18), sep=""))]

agg <- aggregate(. ~ User_ID+Gender+Age+Occupation+City_Category+Stay_In_Current_City_Years+Marital_Status, data, FUN = sum)
agg.count <- aggregate(Product_ID ~ User_ID+Gender+Age+Occupation+City_Category+Stay_In_Current_City_Years+Marital_Status, b, FUN = length)

agg$count <- agg.count$Product_ID
agg$Purchase_avg <- agg$Purchase/agg$count

#### Split data into training and test ####
#f = b[which(b$Gender == "F"),]
#m = b[which(b$Gender == "M"),]
#train_ind_f <- sample(seq_len(nrow(f)),size = floor(nrow(f)*0.7))
#train_ind_m <- sample(seq_len(nrow(m)),size = floor(nrow(m)*0.7))
#BF_train <- rbind(f[train_ind_f,],m[train_ind_m,])
#BF_train <- cbind(BF_train[,-9],BF_train[,9])
#colnames(BF_train)[27] = "Purchase"
#BF_test <- rbind(f[-train_ind_f,],m[-train_ind_m,])
#BF_test <- cbind(BF_test[,-9],BF_test[,9])
#colnames(BF_test)[27] = "Purchase"
#write.csv(BF_train,"BF_train.csv",row.names=TRUE)
#write.csv(BF_test,"BF_test.csv",row.names=TRUE)
BF_train = read.csv("BF_train.csv",header = T)
BF_test = read.csv("BF_test.csv",header = T)

### convert character to numeric ###
trans = function(x){as.numeric(as.factor(x))}
BF_train[,3:8] <- apply(BF_train[,3:8],2,trans)
BF_test[,3:8] <- apply(BF_test[,3:8],2,trans)

attach(BF_train)

###################################################################
#### Visualization ####
attach(agg)
f = agg[which(agg$Gender == "F"),]
m = agg[which(agg$Gender == "M"),]

barplot(rbind(apply(m[,9:26],2,sum)/nrow(m),
              apply(f[,9:26],2,sum)/nrow(f)),col = c("blue","red"))
legend("topright",c("male","female"),col = c("blue","red"),pch = 19)


##################################################################
#### KNN Regression ####
library(FNN)

ptm <- proc.time()
MSE_knn = c()
for(k in 1:20){
  knn = knn.reg(BF_train[,3:26],BF_test[,3:26],BF_train[,27],k=k)
  MSE_knn[k] = mean((knn$pred - BF_test[,27])^2)
}
ptm_knn = proc.time() - ptm

plot(1:20,MSE_knn,type = "b",xlab = "k",ylab = "MSE",main = "MSE of KNN",pch = 19)
abline(v = 6,lty = 5,col = "red")

##################################################################
#### Dimension Reduction ####
library(glmnet)
### Lasso ####
set.seed (2)
lasso.cv.out = cv.glmnet(as.matrix(BF_train[,-c(1,2,27)]),BF_train[,27],alpha =1)
plot(lasso.cv.out)
bestlambda = lasso.cv.out$lambda.min
grid = 10^seq(10,-2, length = 100)
out = glmnet(as.matrix(BF_train[,-c(1,2,27)]),BF_train[,27],
             alpha = 1, lambda = grid)
lasso.coef = predict(out ,type ="coefficients",s=bestlambda)
lasso.coef


##################################################################
#### SVR ####
library(e1071)

t = BF_test[sample(nrow(BF_test),10000),-c(1,2)]
tr = BF_train[sample(nrow(BF_train),10000),-c(1,2)]
# tuning parameter C, higher cost cause harder margin
MSE_linearsvm_c = c()
for(c in c(1:100)){
  svm_linear <- svm(Purchase ~ .,data = tr,
                    kernel="linear",cost = c,scale = F)
  svm.pred <- predict(svm_linear,t[,-25])
  MSE_linearsvm_c[c] = mean((svm.pred - t[,25])^2)
}

plot(1:100,MSE_linearsvm_c,xlab = "cost",ylab = "validation MSE",main = "Linear SVM",pch = 19)
abline(v = 25,col = "red",lty = 5)

#### tuning parameter gamma & cost
#obj_svm = tune.svm(Purchase ~ .,data = tr, gamma = seq(0,1,0.01),cost = c(1:100))
#plot(obj_svm)
#plot(obj_svm, type = "perspective", theta = 50, phi = 20) # theta & phi is angle of the 3D-plot
#svm_para = obj_svm$best.parameters


#### SVR with linear kernel ####
ptm <- proc.time()
ksvm_lin <- svm(Purchase ~ .,data = tr,kernel="linear",
                cost = 25,scale = F)
ptm.ksvm_lin = proc.time() - ptm
ksvm.pred_lin <- predict(ksvm_lin,t[,-25])
MSE_ksvm_lin = mean((ksvm.pred_lin - t[,25])^2)

#### KSVR with Gaussian kernel ####

### tuning parameter by parallel ###
pkgs <- c('foreach', 'doParallel')
lapply(pkgs, require, character.only = T)

## open parallel cores ##
cl <- makeCluster(detectCores()-1)
registerDoParallel(cl)

## PARAMETER LIST ##
parms <- expand.grid(cost = c(1:30), gamma = seq(0,1,0.1))

## LOOP THROUGH PARAMETER VALUES ##
result <- foreach(i = 1:nrow(parms), .combine = rbind) %dopar% {
  c <- parms[i, ]$cost
  g <- parms[i, ]$gamma
  mdl <- e1071::svm(Purchase ~ .,data = tr, kernel = "radial", cost = c, gamma = g, probability = TRUE)
  pred <- predict(mdl,t[,-25],decision.values = TRUE, probability = TRUE)
  m = mean((pred - t[,25])^2)
  data.frame(parms[i, ], mean = m)
}

stopCluster(cl)

### predict with the tuned parameter ###
ptm <- proc.time()
ksvm_gau <- svm(Purchase ~ .,data = tr,kernel="radial",
                cost = 25,gamma = 0.1,scale = F)
ksvm.pred_gau <- predict(ksvm_gau,t[,-25])
MSE_ksvm_gau = mean((ksvm.pred_gau - t[,25])^2)
ptm.ksvm_gau <- proc.time()-ptm

#### KSVR with Sigmoid kernel ####
ksvm_sig <- svm(Purchase ~ .,data = tr,kernel="sigmoid",
                cost = svm_para$cost,gamma = svm_para$gamma,scale = F)
ksvm.pred_sig <- predict(ksvm_sig,t[,-29])
MSE_ksvm_sig = mean((ksvm.pred_sig - t[,29])^2)


#### KSVR with 2nd order polynomial kernel ####
ksvm_poly <- svm(Purchase ~ .,data = tr,kernel="polynomial",
                 cost = svm_para$cost,gamma = svm_para$gamma,scale = F)
ksvm.pred_poly <- predict(ksvm_poly,t[,-29])
MSE_ksvm_poly = mean((ksvm.pred_poly - t[,29])^2)

##################################################################
#### Random Forest ####
require(randomForest)

obj_rf = tune.randomForest(Purchase ~ .,data = BF_train[,-c(1:2)])

rf <- randomForest(Purchase ~ .,data = BF_train[,-c(1:2)],ntree = 20)
rf.pred <- predict(rf, BF_test)
MSE_RF = mean((rf.pred - BF_test[,29])^2)

##################################################################
#### Kmeans ####
require(factoextra)

agg_clus = agg[,-c(1:8,29,30)]

fviz_nbclust(agg_clus, 
             FUNcluster = kmeans,# K-Means
             method = "wss",     # total within sum of square
             k.max = 20          # max number of clusters to consider
) +
  
  labs(title="Elbow Method for K-Means") +
  
  geom_vline(xintercept = 5,        
             linetype = 2)          

# 5 clusters
kmeans.cluster <- kmeans(agg_clus, centers=5)

# variance within groups
kmeans.cluster$withinss

# visualization of Kmeans
fviz_cluster(kmeans.cluster,          
             data = agg_clus,            
             geom = c("point","text"),
             frame.type = "norm")

