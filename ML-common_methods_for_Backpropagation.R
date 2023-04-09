## Common methods for Backpropagation

softmax <- function(a){
  max.a <- max(a)
  return(exp(a-max.a)/sum(exp(a-max.a)))
}

cross_entropy_error <- function(y,t){
  # yは行ごとに出力がまとめられたものとする
  batch_size <- nrow(y)
  delta <- matrix(1e-7,nrow(y),ncol(y))
  return(-sum(t * log(y + delta))/batch_size)
}


setClass("MulLayer",
         representation(x="numeric",
                        y="numeric"),
         prototype = list(x=NULL, y=NULL))

setGeneric("forward",
           def = function(object,...){
             object
           })
setMethod("forward","MulLayer",
          function(object){
            return(object@x*object@y)
          })

setGeneric("backward",
           def = function(object,dout){
             object
           })
setMethod("backward","MulLayer",
          function(object, dout){
            dx <- dout * object@y
            dy <- dout * object@x
            return(list(dx=dx,dy=dy))
          })


#activation functions


setClass("Relu",
         representation(mask="matrix"))

setMethod("forward","Relu",
          function(object,x){
            object@mask <- (x<=0)
            out <- x
            out[object@mask] <- 0
            return(list(object,out))
          })

setMethod("backward","Relu",
          function(object,dout){
            dout[object@mask] <- 0
            dx <- dout
            return(list(object,dout))
          })


# xはダミー
setClass("Sigmoid",
         representation(out="matrix"))

setMethod("forward","Sigmoid",
          function(object,x){
            out <- 1 / (1 + exp(-x))
            object@out <- out
            return(list(object,out))
          })

setMethod("backward","Sigmoid",
          function(object,dout){
            dx <- dout * (1.0 - object@out) * object@out
            return(dx)
          })

setClass("Affine",
         representation(W="matrix",
                        b="matrix",
                        x="matrix",
                        dW="matrix",
                        db="matrix"))

setMethod("forward","Affine",
          function(object,x){
            object@x <- x
            out <- x %*% object@W + object@b
            return(list(object,out))
          })

setMethod("backward","Affine",
          function(object,dout){
            dx <- dout %*% t(object@W)
            object@dW <- t(object@x) %*% dout
            # applyではなくcolsumでもできる
            #object@db <- apply(dout,2,sum) \ エラーになったのでmatrixに変換
            object@db <- t(matrix(colSums(dout)))
            return(list(object,dx))
          })

#5.6.3

setClass("SoftmaxWithLoss",
         representation(loss="numeric",
                        y="matrix",
                        t="matrix"))

setMethod("forward","SoftmaxWithLoss",
          function(object,x,t){
            #object = lastlayer
            object@t <- t
            
            #softmax
            max.x = apply(x,1,max)
            x = x - max.x %*% matrix(1,1,ncol(x))
            y = exp(x)/rowSums(exp(x))
            object@y = y
            
            #loss
            delta = 1e-7
            R = nrow(as.matrix(y))
            object@loss = -sum(t * log(y + delta))/R
            return(object)
          })

setMethod("backward","SoftmaxWithLoss",
          function(object,dout=1){
            batch_size <- nrow(object@t)
            dx <- (object@y - object@t) / batch_size
            return(dx)
          })



## TwoLayerNet

setClass("TwoLayerNet",
         representation(params="list",
                        layers="list",
                        lastlayer="SoftmaxWithLoss",
                        input_size="numeric",
                        hidden_size="numeric",
                        output_size="numeric",
                        weight_init_std="numeric"),
         prototype = list(weight_init_std = 0.01,
                          params = list(W1 = NULL,W2 = NULL,B1 = NULL,B2 = NULL))
)

# 初期化
setMethod("initialize","TwoLayerNet",
          function(.Object, ...){
            .Object <- callNextMethod()
            # 重みの初期化
            .Object@params$W1 <- matrix(rnorm(.Object@input_size * .Object@hidden_size),
                                        .Object@input_size, .Object@hidden_size) * .Object@weight_init_std
            .Object@params$W2 <- matrix(rnorm(.Object@hidden_size * .Object@output_size),
                                        .Object@hidden_size, .Object@output_size) * .Object@weight_init_std
            .Object@params$B1 <- rbind(rep(0,.Object@hidden_size))
            .Object@params$B2 <- rbind(rep(0,.Object@output_size))
            
            # レイヤ生成
            #.Object@layers <- list("Affine1","Relu1","Affine2")
            Affine1 <- new("Affine", W=.Object@params$W1, b=.Object@params$B1)
            Relu <- new("Relu")
            Affine2 <- new("Affine", W=.Object@params$W2, b=.Object@params$B2)
            .Object@layers <- list(Affine1,Relu,Affine2)
            
            .Object@lastlayer <- new("SoftmaxWithLoss")
            
            .Object
          })

setMethod("predict",
          signature(object="TwoLayerNet"),
          function(object,x){
            for(layer in 1:length(object@layers)){
              res <- forward(object@layers[[layer]],x)
              object@layers[[layer]] <- res[[1]]
              x <- res[[2]]
            }
            return(list(object,x))
          })

setGeneric("loss",
           def = function(object,x,t){
             object
           })
setMethod("loss",
          signature(object="TwoLayerNet"),
          function(object,x,t){
            res <- predict(object,x)
            object <- res[[1]]
            y <- res[[2]]
            
            object@lastlayer <- forward(object@lastlayer,y,t)
            
            return(object)
          })

setGeneric("accuracy",
           def = function(object,x,t){
             object
           })
setMethod("accuracy",
          signature(object="TwoLayerNet"),
          function(object,x,t){
            res <- predict(object,x)
            y <- res[[2]]
            
            total <- nrow(y)
            correct <- 0
            # 行ごとに最大値の位置を比較
            for( i in 1:nrow(y)) {
              if(which.max(y[i,]) == which.max(t[i,])){
                correct <- correct + 1
              }
            }
            return(correct / total)
          })

setGeneric("gradient",
           def = function(object,...){
             object
           })
setMethod("gradient",
          signature(object="TwoLayerNet"),
          function(object,x,t){
            # forward
            object <- loss(object,x,t)
            # backward
            dout <- 1
            dout <- backward(object@lastlayer)

            for(layer in length(object@layers):1){
              res <- backward(object@layers[[layer]],dout)
              object@layers[[layer]] <- res[[1]]
              dout <- res[[2]]
            }
            
            grads <- list(W1<-object@layers[[1]]@dW,
                          W2<-object@layers[[3]]@dW,
                          b1<-object@layers[[1]]@db,
                          b2<-object@layers[[3]]@db)
            
            return(list(object,grads))
          })
