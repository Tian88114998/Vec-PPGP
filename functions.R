library(Rcpp)
library(RcppEigen)
sourceCpp(file='demo_code_W2/src/demo.cpp') 
n=10^8


###1. comparing time for one loop
#C++
time1=system.time(
  for(i in 1:1){
    ans1=Test_for_loop(n)
  }
)
#R
ans2=0
time2=system.time(
  for(i in 1:n){
    ans2=ans2+0.1
  }
)
time1
time2

##2. Cholesky
n=600
x=seq(0,1,1/(n-1))
R0=abs(outer(x,(x),'-'))
##the larger the gamma, the more singular the matrix is  
gamma=.1
R=exp(-(R0/gamma)^{1.9})

#C++
system.time(
  for(i in 1:1){
   L1=Chol(R)
  }
)
##R
system.time(
  for(i in 1:1){
    L2=t(chol(R))
  }
)
max(abs(L1-L2)) ##difference

###R_inv y
y=runif(n)
##C++
system.time(
  for(i in 1:1){
    R_inv_y_1=R_inv_y(R, y)
  }
)
##R
system.time(
  for(i in 1:1){
    L2=t(chol(R))
    R_inv_y_2=(backsolve(t(L2), forwardsolve(L2,y)))
  }
)

#direct inverse
system.time(
  for(i in 1:1){
    R_inv_y_3=solve(R)%*%y 
  }
)


