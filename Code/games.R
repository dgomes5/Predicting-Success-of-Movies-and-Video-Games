# John Gomes
# Daniel Gomes
# Blake Simmons
# CIS 490 Final Project


set.seed(490)
library(readr)
library(ridge)
library(rpart)
library(rpart.plot)
library(randomForest)

#Videogame sales dataset
games <- read.csv("video-game-sales-with-ratings/Video_Games_Sales_as_at_22_Dec_2016.csv")

names(games) <- c('NAME', 'PLAT', 'YOR', 'GEN', 'PUB', 'NA_S', 'EU_S', 'JP_S', 'OT_S', 'GL_S', 'CR_S', 'CR_C', 'UR_S', 'UR_C', 'DEV', 'RATE')
library(glmnet)
summary(games)

sapply(games, class)
as.factor(games$RATE)
games$UR_S <- as.numeric(games$UR_S)
games$CR_S <- as.numeric(games$CR_S)
games$CR_C <- as.numeric(games$CR_C)
games$UR_C <- as.numeric(games$UR_C)

games <- as.data.frame(games)

drops <- c("NAME")
games <- games[ , !(names(games) %in% drops)]

games <- na.omit(games)

games$CR_S <- as.integer(games$CR_S)

frml <- GL_S ~ PLAT + YOR + GEN + RATE + CR_S + CR_C + UR_S + UR_C + NA_S #+ EU_S + JP_S + OT_S  
training.indices <- sample(1:nrow(games), size=nrow(games) * 0.8)
#training.indices <- sample(1:nrow(games), 0.8 * nrow(games))

games.mat <- model.matrix(frml, data = games)

sapply(games, class)

X <- games.mat
Y <- games[,"GL_S"]

X.train <- X[training.indices,]
X.test <- X[-training.indices,]
Y.train <- Y[training.indices]
Y.test <- Y[-training.indices]

lasso.mod <- glmnet(X.train, Y.train)
plot(lasso.mod, xvar="lambda")

lasso.cv.out <- cv.glmnet(X, Y, alpha=1, nfolds = 10)

plot(lasso.cv.out)

lasso.mod.pred <- predict(lasso.mod, X.test)
rmse <- sqrt(apply((lasso.mod.pred - Y.test)^2, 2, mean))
lasso.best.lambda <- lasso.cv.out$lambda.1se
print(lasso.best.lambda)

lasso.pred <- predict(lasso.mod, s=lasso.best.lambda, newx=X.test)
print(paste('RMSE:', sqrt(mean((lasso.pred - Y.test)^2))))

lasso.games.rmse <- sqrt(mean((lasso.pred - Y.test)^2))

lasso.best.lambda

grid <- 10^seq(10, -2, length=1000)

lasso.out <- glmnet(X, Y, alpha=1, lambda=grid)
lasso.coef <- predict(lasso.out, type="coefficients", s=lasso.best.lambda)
lasso.coef



#Begin Regression Tree (Random Forest)

#training.indices <- sample(1:nrow(games), size=nrow(games) * 0.7)
games.train <- games[training.indices,]
games.test <- games[-training.indices,]

rf.games <- randomForest(frml, data=games.train, mtry=round(sqrt(ncol(games.train) -1)), importance=TRUE)
varImpPlot(rf.games, cex=0.5)
plot(rf.games)

#trees = 150

p <- ncol(games.train) - 1
oob.error <- double(p)
test.error <- double(p)
set.seed(3)

for(m in 1:p) {
  fit <- randomForest(frml, data=games.train, mtry=m, ntree=150)
  oob.error[m] <- fit$mse[150]
  test.error[m] <- mean((games.test$GL_S - predict(fit, newdata=games.test))^2)
}

matplot(1:p, cbind(oob.error, test.error), pch=19, col=c("red", "blue"), type="b", ylab="Mean Squared Error")
legend("top", c('OOB', 'Test'), col=seq_len(2), cex=0.8, fill=c("red", "blue"))

rf.pred <- predict(rf.games, newdata = games.test)
library(caret)

rf.games.rmse <- RMSE(rf.pred, games.test$GL_S)

print(paste("Lasso RMSE: ", lasso.games.rmse))
print(paste("Random Forest RMSE: ", rf.games.rmse))

plot(rf.pred)
points(games.test$GL_S, col=2)

plot(lasso.pred)
points(Y.test, col=2)
