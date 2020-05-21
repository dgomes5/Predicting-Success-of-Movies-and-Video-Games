# Blake Simmons, Daniel Gomes, John Gomes
# CIS490
# Final Project

library(ridge)
library(glmnet)
library(ggplot2)
library(jtools)
library(readr)
library(rpart)
library(rpart.plot)
library(randomForest)
options(scipen = 999)

#import data
movies <- read_csv("tmdb-5000-movie-dataset/tmdb_5000_movies.csv")
credits <- read_csv("tmdb-5000-movie-dataset/tmdb_5000_credits.csv")

#clean data, drop columns with empty "cast" or "crew"
drop.rows <- c(2602, 3662, 3671, 3972, 3978, 3993, 4010, 4069, 4106, 4119, 4124, 4248, 4294, 4306, 
               4315, 4323, 4386, 4401, 4402, 4406, 4414, 4432, 4459, 4492, 4505, 4509, 4518, 4551, 
               4554, 4563, 4565, 4567, 4570, 4572, 4582, 4582, 4584, 4590, 4612, 4617, 4618, 4623, 
               4634, 4639, 4639, 4645, 4658, 4663, 4675, 4680, 4682, 4686, 4690, 4699, 4711, 4713, 
               4715, 4717, 4738, 4758, 4756, 4798, 4802)
movies <- movies[-c(drop.rows), ]
credits <- credits[-c(drop.rows), ]



#JSON processing
library(jsonlite)

#cast from JSON (credits$cast)
cast <- purrr::map(credits$cast, jsonlite::fromJSON)

#crew from JSON (credits$crew)
crew <- purrr::map(credits$crew, jsonlite::fromJSON)

#create list of "stars"
starring <- vector("character", length(cast))
index <- 1

#create list of "name" of first actor in each movie
for (i in cast) {
  #print(i[1,6])
  starring[[index]] <- i[1,6]
  index <- index + 1
}

#add "starring" column to movies dataframe
movies$starring <- starring





#create list of "directors"
director <- vector("character", length(crew))
index <- 1

#create list of "names" that correspond with rows that contain the "job" Director
for (i in crew) {
  director[[index]] <-i[,6][which( i["job"] == "Director" )][1]
  index <- index + 1
}

#add director column to movies dataframe
movies$director <- director




#genre from JSON 
genres <- purrr::map(movies$genres, jsonlite::fromJSON)

genre <- vector("character", length(genres))
index <- 1

for (i in genres) {
  #print(i[1,6])
  genre[[index]] <- i[1,2]
  index <- index + 1
}

movies$genre <- genre




#drop unneccessary columns to make things easier
drops <- c("title", "genres","homepage", "id", "keywords", "overview", "status", "tagline", "spoken_languages", "original_title")
movies <- movies[ , !(names(movies) %in% drops)]

#further cleaning, drop rows with empty JSON and rows with budget and revenue < 30 (clear errors)
movies <- movies[apply(movies[c(1)],1,function(z) !any(z<10000)),]
movies <- movies[apply(movies[c(7)],1,function(z) !any(z<10000)),]


movies <- movies[apply(movies[c(4)],1,function(z) !any(z=="[]")),]
movies <- movies[apply(movies[c(5)],1,function(z) !any(z=="[]")),]


#create production company column from JSON
productionco <- purrr::map(movies$production_companies, jsonlite::fromJSON)

production <- vector("character", length(productionco))
index <- 1

for (i in productionco) {
  #print(i[1,6])
  production[[index]] <- i[1,1]
  index <- index + 1
}

movies$production_companies <- production



#create production country column from JSON

country <- purrr::map(movies$production_countries, jsonlite::fromJSON)

prod.country <- vector("character", length(country))
index <- 1

for (i in country) {
  #print(i[1,6])
  prod.country[[index]] <- i[1,1]
  index <- index + 1
}

movies$production_countries <- prod.country

#scale revenue and budget data to make numbers more manageable (in millions)
movies$budget <- movies$budget/1000000
movies$revenue <- movies$revenue/1000000



#observing and reassigning column datatypes
sapply(movies, class)
movies$starring <- as.factor(movies$starring)
movies$director <- as.factor(movies$director)
movies$original_language <-as.factor(movies$original_language)
movies$production_companies <- as.factor(movies$production_companies) 
movies$production_countries <- as.factor(movies$production_countries) 
movies$genre <- as.factor(movies$genre)

movies <- as.data.frame(movies)

#begin regression
frml <- revenue ~ budget + original_language + popularity + production_countries + release_date + runtime + vote_average + vote_count + genre
set.seed(86)
#training.indices <- sample(1:nrow(movies), 0.8 * nrow(movies))
training.indices <- sample(1:nrow(movies), size=nrow(movies) * 0.8)

movies.mat <- model.matrix(frml, data = movies)

X <- movies.mat
Y <- movies[,"revenue"]

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

lasso.movies.rmse <- sqrt(mean((lasso.pred - Y.test)^2))

lasso.best.lambda

grid <- 10^seq(10, -2, length=1000)

lasso.out <- glmnet(X, Y, alpha=1, lambda=grid)
lasso.coef <- predict(lasso.out, type="coefficients", s=lasso.best.lambda)
lasso.coef



#Begin Regression Tree (Random Forest with Boosting)


#training.indices <- sample(1:nrow(movies), size=nrow(movies) * 0.8)


movies.train <- movies[training.indices,]
movies.test <- movies[-training.indices,]

rf.movies <- randomForest(frml, data=movies.train, mtry=round(sqrt(ncol(movies.train) -1)), importance=TRUE)
varImpPlot(rf.movies, cex=0.5)
plot(rf.movies)

#trees = 200

ncol(movies.train) -1
p <- ncol(movies.train) - 1
oob.error <- double(p)
test.error <- double(p)
#set.seed(3)

for(m in 1:p) {
  fit <- randomForest(frml, data=movies.train, mtry=m, ntree=200)
  oob.error[m] <- fit$mse[200]
  test.error[m] <- mean((movies.test$revenue - predict(fit, newdata=movies.test))^2)
}

matplot(1:p, cbind(oob.error, test.error), pch=19, col=c("red", "blue"), type="b", ylab="Mean Squared Error")
legend("top", c('OOB', 'Test'), col=seq_len(2), cex=0.8, fill=c("red", "blue"))

rf.pred <- predict(rf.movies, newdata = movies.test)
library(caret)

rf.movies.rmse <- RMSE(rf.pred, movies.test$revenue)

print(paste("Lasso RMSE: ", lasso.movies.rmse))
print(paste("Random Forest RMSE: ", rf.movies.rmse))

range(Y.test)
range(movies.test$revenue)
