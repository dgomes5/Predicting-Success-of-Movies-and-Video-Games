# John Gomes
# Daniel Gomes
# Blake Simmons
# CIS 490 Final Project


set.seed(490)
library(readr)
library(ridge)
#Videogame sales dataset
games <- read.csv("Video_Games_Sales_as_at_22_Dec_2016.csv")

names(games) <- c('NAME', 'PLAT', 'YOR', 'GEN', 'PUB', 'NA_S', 'EU_S', 'JP_S', 'OT_S', 'GL_S', 'CR_S', 'CR_C', 'UR_S', 'UR_C', 'DEV', 'RATE')
library(glmnet)
summary(games)

games <- as.data.frame(games)

frml <- GL_S ~ .
set.seed(7)
nrow(games)
training.indices <- sample(1:nrow(games), 0.8 * nrow(games))
games.mat <- model.matrix(frml, data = games)

X <- games.mat
Y <- games[, "GL_S"]

X.train <- X[training.indices,]
X.test <- X[-training.indices,]
Y.train <- Y[training.indices]
Y.test <- Y[-training.indices]

lasso.mod <- glmnet(X.train, Y.train)
plot(lasso.mod, xvar="lambda")
lasso.cv.out <- cv.glmnet(X, Y, alpha=1, nfolds = 10)
plot(lasso.cv.out)

lasso.mod.pred <- predict(lasso.mod, as.matrix(X.test))
rmse <- sqrt(apply((lasso.mod.pred - Y.test)^2, 2, mean))
lasso.best.lambda <- lasso.cv.out$lambda.1se
print("Lasso Regression RMSE:")
print(lasso.best.lambda)
lasso.pred <- predict(lasso.mod, s=lasso.best.lambda, newx=as.matrix(X.test))
print(paste('RMSE:', sqrt(mean((lasso.pred - Y.test)^2))))

lasso.out <- glmnet(as.matrix(X), Y, alpha=1)
lasso.coef <- predict(lasso.out, type="coefficients", s=lasso.best.lambda)
lasso.coef

lasso.mod <- glmnet(as.matrix(X.train), Y.train)
plot(lasso.mod, xvar="lambda")
lasso.cv.out <- cv.glmnet(as.matrix(X), Y, alpha=1, nfolds = 10)
plot(lasso.cv.out)

print(paste('Lasso Regression RMSE:',lasso.best.lambda ))
