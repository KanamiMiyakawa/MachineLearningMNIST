#MNISTデータセットからニューラルネットワークを用いて手書き文字の学習を行う

library(dplyr)

source("ML-common_methods_for_Backpropagation.R")

#データセットとハイパーパラメータ
#data_set <- "iris"
data_set <- "MNIST"

if(data_set=="iris"){
  # iris
  iris.x.all = as.matrix(iris[,1:4])
  iris.y.temp = as.numeric(iris[,5])
  iris.y.all = matrix(0,nrow = nrow(iris.x.all), ncol =3)
  iris.y.all[which(iris.y.temp==1), 1]=1
  iris.y.all[which(iris.y.temp==2), 2]=1
  iris.y.all[which(iris.y.temp==3), 3]=1
  
  # テストデータと訓練データに分割
  df.rows = nrow(iris.x.all)
  train.rate = 0.7
  train.index <- sample(df.rows, df.rows * train.rate)
  train.x = iris.x.all[train.index,]
  train.y = iris.y.all[train.index,]
  test.x = iris.x.all[-train.index,]
  test.y = iris.y.all[-train.index,]
  
  # ハイパーパラメータ
  input_size = 4
  hidden_size = 15
  output_size = 3

  iters_num <- 1000
  train_size <- nrow(train.x)
  batch_size <- 15
  learning_rate <- 0.1
}else if(data_set=="MNIST"){
  # MNIST
  MNIST <- read.csv("MNSTtrain.csv", 
                    header=TRUE)
  MNIST <- data.matrix(MNIST)
  MNIST.x.all <- MNIST[,-1]
  MNIST.x.all <- MNIST.x.all/255
  
  # 正解データを成形
  MNIST.y.val <- MNIST[,1] + 1
  MNIST.y.all <- matrix(0,nrow(MNIST.x.all),10)
  for(i in 1:nrow(MNIST.x.all)){
    MNIST.y.all[i,MNIST.y.val[i]] <- 1
  }
  
  # テストデータと訓練データに分割
  df.rows = nrow(MNIST.x.all)
  train.rate = 0.7
  train.index <- sample(df.rows, df.rows * train.rate)
  train.x = MNIST.x.all[train.index,]
  train.y = MNIST.y.all[train.index,]
  test.x = MNIST.x.all[-train.index,]
  test.y = MNIST.y.all[-train.index,]
  
  # ハイパーパラメータ
  input_size = 784
  hidden_size = 50
  output_size = 10
  
  iters_num <- 2000
  train_size <- nrow(train.x)
  batch_size <- 200
  # learning_rateはスケールによって異なる
  learning_rate <- 0.1
}



#インスタンス作成
network <- new("TwoLayerNet",
               input_size=input_size, hidden_size=hidden_size, output_size=output_size)

train_loss_list <- vector()
train_acc_list <- vector()
test_acc_list <- vector()

# validationの上がり始めたところをエポックの区切りにすればいい
# なので最初は多めに設定して学習し直す
iter_per_epoch <- round(max(train_size / batch_size,1))

# バッチサイズに合わせてB1,B2を成形
network@params[[3]] <- matrix(rep(network@params[[3]],batch_size),batch_size,hidden_size)
network@layers[[1]]@b <- network@params[[3]]
network@params[[4]] <- matrix(rep(network@params[[4]],batch_size),batch_size,output_size)
network@layers[[3]]@b <- network@params[[4]]


for(i in 1:iters_num){

  # ミニバッチの成形
  batch_mask <- sample(1:train_size,batch_size)
  x_batch <- train.x[batch_mask,]
  y_batch <- train.y[batch_mask,]
  
  # 勾配計算
  res <- gradient(network,x_batch,y_batch)
  network <- res[[1]]
  grads <- res[[2]]
  
  # bをバッチサイズに合わせて成形
  grads[[3]] <- matrix(rep(grads[[3]],batch_size),batch_size,byrow=T)
  grads[[4]] <- matrix(rep(grads[[4]],batch_size),batch_size,byrow=T)
  
  # パラメータ更新
  for(idx in 1:4){
    network@params[[idx]] <- network@params[[idx]] - learning_rate * grads[[idx]]
  }
  
  #レイヤ更新
  network@layers[[1]]@W <- network@params[[1]]
  network@layers[[3]]@W <- network@params[[2]]
  network@layers[[1]]@b <- network@params[[3]]
  network@layers[[3]]@b <- network@params[[4]]
  
  # 損失を記録
  loss_val <- loss(network, x_batch, y_batch)
  train_loss_list <- c(train_loss_list, loss_val@lastlayer@loss)
  
  # エポック処理
  if(i %% iter_per_epoch == 0){
    acc.train <- accuracy(network, x_batch, y_batch)
    train_acc_list <- c(train_acc_list, acc.train)
    
    # テストサイズに合わせてbを成形
    network.test <- network
    network.test@layers[[1]]@b <- matrix(rep(network@params[[3]][1,],nrow(test.x)),
                                         nrow(test.x),byrow=TRUE)
    network.test@layers[[3]]@b <- matrix(rep(network@params[[4]][1,],nrow(test.x)),
                                         nrow(test.x),byrow=TRUE)
    
    acc.test <- accuracy(network.test, test.x, test.y)
    test_acc_list <- c(test_acc_list, acc.test)
  }
}

plot(train_loss_list, type="l", xlab="batch count", ylab="loss")
par(new=T)
plot(train_acc_list, type="l", lty=6, xlab ="", ylab="", col="blue",
     xaxt="n",yaxt="n")
par(new=T)
plot(test_acc_list, type="l", lty=6, xlab="", ylab="", col="red",
     xaxt="n",yaxt="n")
legend("topright", inset = c(0, 0.5), 
       legend=c("loss", "train accuracy", "test accuracy"), 
       col=c("black", "red", "blue"), lty=1, box.lty=0)


