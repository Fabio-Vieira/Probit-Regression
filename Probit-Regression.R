#Probit Regression with a real data application using the credit card applications
#data set from the UCI repository
###################################################################################
####Simulation
N <- 100
x <- cbind(rep(1,N),seq(-1,1,length.out=N))
beta <- c(-0.5,3)
prob <- pnorm(x %*% beta)

y <- rbinom(N,1,prob)

##################################################################################
###############################################################################
library(MASS) #This package needs to be load, so we can sample from a multivatiate
#normal distribution
library(truncnorm)


updateBeta <- function(Sig2Be,X,Y,Sig2){ 
  #This function provides a Gibbs sampler for the regression parameters, 
  # and it can adapt to simple and multiple linear regression, all that is
  #need is to adjust the dimension of the design matrix
  n <- dim(X)[2]
  B0 <- diag(Sig2Be,n,n)
  Sigma <- Sig2 * diag(1,ncol=n,nrow=n)
  sig2.post <- solve((t(X)%*%X)%*%Sigma + B0)
  m.post <- (t((t(X)%*%Y))%*%Sigma) %*% sig2.post
  
  return(mvrnorm(1,m.post,sig2.post))
}

updateYstar <- function(Y,Beta,X){
  Ystar <- array(NA,dim=length(Y))
  m.latent <- X%*%Beta
  n_suc <- sum(Y)
  n_fails <- length(Y) - sum(Y)
  Ystar[Y==0] <- rtruncnorm(n_fails,mean=m.latent[Y==0],sd=1,a=-Inf,b=0)
  Ystar[Y==1] <- rtruncnorm(n_suc,mean=m.latent[Y==1],sd=1,a=0,b=Inf)
  return(Ystar)
}

###################################################################################
Niter <- 20000
Y <- y
X <- x
Beta.out <- array(NA, dim = c(Niter,dim=dim(X)[2]))

Beta.out[1,] <- beta

for(i in 2:Niter){
  Ystar <- updateYstar(Y,Beta.out[i-1,],X)
  Beta.out[i,] <- updateBeta(0.0001,X,Ystar,1)
  print(i)
}

plot(Beta.out[,1],type='l')
abline(h=beta[1],col='red')
hist(Beta.out[,1])
abline(v=quantile(Beta.out[,1],probs=c(0.025)),lty=2,col='blue')
abline(v=quantile(Beta.out[,1],probs=c(0.975)),lty=2,col='blue')
abline(v=beta[1],col='red')
mean(Beta.out[,1])

plot(Beta.out[,2],type='l')
abline(h=beta[2],col='red')
hist(Beta.out[,2])
abline(v=quantile(Beta.out[,2],probs=c(0.025)),lty=2,col='blue')
abline(v=quantile(Beta.out[,2],probs=c(0.975)),lty=2,col='blue')
abline(v=beta[2],col='red')
mean(Beta.out[,2])

Nburn <- 1001
post_est <- colMeans(Beta.out[-c(1:Nburn),])
plot(x[,2],y,main="Simulation")
lines(x=x[,2],y=pnorm(X%*%post_est),col="red",lwd=2)

###################################################################################
####Application
set.seed(24)
#Preparing the data, it is the credit approval data set from the UCI repository
data <- as.data.frame(read.csv("crxdata.csv",sep=",",header = F))
Y <- as.factor(data[,16])
Y <- ifelse(Y=="-",0,1)
#I have chosen the variables that seemed to have the biggest predictive power
V9 <- cbind(ifelse(data[,9] == "f",1,0),ifelse(data[,9]== "t",1,0))
V10 <- cbind(ifelse(data[,10] == "f",1,0),ifelse(data[,10]== "t",1,0))
V13 <- cbind(ifelse(data[,13] == "g",1,0),ifelse(data[,13]== "p",1,0),
            ifelse(data[,13] == "s",1,0))
Data <- cbind(Y,V9,V10,V13)
library(caTools)
#Randomly spliting the data into a fitting and a testing set.
divide <- sample.split(Y,SplitRatio = 0.85)
fitting <- subset(Data,divide == T)
testing <- subset(Data,divide == F)
##################################################################################
#fitting the model with MCMC
Y1 <- fitting[,1]
X1 <- cbind(rep(1,length(Y1)),fitting[,2:8])

Niter <- 50000
Beta.out <- array(NA, dim = c(Niter,dim=dim(X1)[2]))

Beta.out[1,] <- rep(1,dim(X1)[2])

for(i in 2:Niter){
  Ystar <- updateYstar(Y1,Beta.out[i-1,],X1)
  Beta.out[i,] <- updateBeta(0.0001,X1,Ystar,1)
  print(i)
}

#Checking for convergence 
plot(Beta.out[,1],type='l')
hist(Beta.out[,1])
mean(Beta.out[,1])

plot(Beta.out[,2],type='l')
hist(Beta.out[,2])
mean(Beta.out[,2])

plot(Beta.out[,3],type='l')
hist(Beta.out[,3])
mean(Beta.out[,3])

plot(Beta.out[,4],type='l')
hist(Beta.out[,4])
mean(Beta.out[,4])

plot(Beta.out[,5],type='l')
hist(Beta.out[,5])
mean(Beta.out[,5])

plot(Beta.out[,6],type='l')
hist(Beta.out[,6])
mean(Beta.out[,6])

plot(Beta.out[,7],type='l')
hist(Beta.out[,7])
mean(Beta.out[,7])

plot(Beta.out[,8],type='l')
hist(Beta.out[,8])
mean(Beta.out[,8])
#################################################################################
Nburn <- 5000 #Burn-in phase
Beta <- Beta.out[-c(1:(Nburn)),]
Y2 <- testing[,1]
X2 <- cbind(rep(1,length(Y2)),testing[,2:8])

Forecasting <- array(NA,dim=c(dim(Beta)[1],length(Y2)))

for(i in 1:nrow(Forecasting)){
  Forecasting[i,] <- X2 %*% Beta[i,] + rnorm(length(Y2))
}

means <- colMeans(Forecasting)
means <- ifelse(means < 0,0,1)
#It has predicted correctly 50 out of 57 Y = 0 and 44 out of 46 Y = 1
table(Y2,means)
