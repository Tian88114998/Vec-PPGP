# import cpp functions
sourceCpp("src/demo.cpp")
################################################################################
### Simulated Dataset
## Vec-PPGP L-BFGS, No Nugget Estimate, Anistropic
m = 25
set.seed(2001)
n = 1500
p = 5
k = 5
temp = generate_data(n, p, k, gamma_range, sigma2_range, nugget_range)
Xo = temp[[1]]
Yo = temp[[2]]
maxmin_order <- GpGp::order_maxmin(Xo)
ordered_sample <- Xo[maxmin_order,]
ordered_response <- Yo[maxmin_order,]

X_train_ordered <- ordered_sample[1:1000,]
X_test_ordered <- ordered_sample[1001:1500,]
Y_train_ordered <- ordered_response[1:1000,]
Y_test_ordered <- ordered_response[1001:1500,]

nnmatrix_all <- GpGp::find_ordered_nn(ordered_sample, m = m)
nnmatrix_train <- GpGp::find_ordered_nn(X_train_ordered, m = m)

set.seed(2001)
gamma = runif(n = p, min = gamma_range[1], max = gamma_range[2])
nu = runif(n = 1, min = nugget_range[1], max = nugget_range[2])
eta = log(1/gamma)
tau = log(nu)
# To train Isotropic, use eta_iso and set isotropic = T
gamma_iso = runif(n = 1, min = gamma_range[1], max = gamma_range[2])
eta_iso = log(1/gamma_iso)
set.seed(2001)
start_time = Sys.time()
res = optim(par = c(eta), fn = neg_vecchia_marginal_log_likelihood, 
            gr = neg_vecchia_marginal_log_likelihood_deriv, nu = 0.2 , 
            nugget_est = FALSE, x = X_train_ordered, NNmatrix = nnmatrix_train, 
            y = Y_train_ordered,  method = "L-BFGS-B", kernel_type = "matern_3_2", 
            alpha = 1.9, trend = matrix(1,1000,1), isotropic = F, zero_mean = "Yes")
end_time = Sys.time()
end_time - start_time 

## Vec-PPGP No Nugget Prediction, Anistropic
gamma_trained = exp(-res$par[1:length(res$par)])
list = predict(x = X_train_ordered, xp = X_test_ordered, NNmatrix = nnmatrix_all, 
               y = Y_train_ordered, gamma = gamma_trained, nu = 0.2, 
               kernel_type = "matern_3_2", alpha = 1.9, q_95 = qnorm(0.975, 0, 1), 
               trend = matrix(1,1000,1), testing_trend = matrix(1,500,1), 
               isotropic = F, zero_mean = "Yes")

mse(Y_test_ordered, list[[1]])
pci(Y_test_ordered, list[[2]], list[[3]])
lci(list[[2]], list[[3]])

## PPGP No Nugget Estimate
start_time <- Sys.time()
m.ppgasp=ppgasp(design=X_train_ordered, response=Y_train_ordered, nugget = 0.2, 
                nugget.est = F, kernel_type = "matern_3_2", isotropic=F, 
                method = "mmle", zero.mean = "Yes", optimization = 'lbfgs')
end_time <- Sys.time()
end_time - start_time

m_pred.ppgasp = RobustGaSP::predict(m.ppgasp, testing_input = X_test_ordered)
mse(Y_test_ordered, m_pred.ppgasp$mean)
pci(Y_test_ordered, m_pred.ppgasp$lower95, m_pred.ppgasp$upper95)
lci(m_pred.ppgasp$lower95, m_pred.ppgasp$upper95)

## Vec-PPGP L-BFGS, With Nugget Estimate, Anistropic
set.seed(2001)
start_time = Sys.time()
res = optim(par = c(eta, tau), fn = neg_vecchia_marginal_log_likelihood, 
            gr = neg_vecchia_marginal_log_likelihood_deriv, method = "L-BFGS-B", 
            nu = nu, nugget_est = TRUE, x = X_train_ordered, NNmatrix = nnmatrix_train, 
            y = Y_train_ordered, kernel_type = "matern_3_2",alpha = 1.9, 
            trend = matrix(1,1000,1), isotropic = F, zero_mean = "Yes")
end_time = Sys.time()
end_time - start_time

gamma_trained = exp(-res$par[1:length(res$par)-1])
nu_trained = exp(res$par[length(res$par)])
list = predict(x = X_train_ordered, xp = X_test_ordered, NNmatrix = nnmatrix_all, y = Y_train_ordered, gamma = gamma_trained, nu = nu_trained, kernel_type = "matern_3_2", alpha = 1.9, q_95 = qnorm(0.975, 0, 1), trend = matrix(1,1000,1), testing_trend = matrix(1,500,1), isotropic = F, zero_mean = "Yes")

mse(Y_test_ordered, list[[1]])
pci(Y_test_ordered, list[[2]], list[[3]])
lci(list[[2]], list[[3]])

## PPGP with Nugget Estimate
start_time <- Sys.time()
m.ppgasp=ppgasp(design=X_train_ordered, response=Y_train_ordered, nugget.est = T, 
                kernel_type = "matern_3_2", isotropic = F, zero.mean = "Yes", 
                method = "mmle", optimization = 'lbfgs')
end_time <- Sys.time()
end_time - start_time

m_pred.ppgasp = RobustGaSP::predict(m.ppgasp, testing_input = X_test_ordered)
mse(Y_test_ordered, m_pred.ppgasp$mean)
pci(Y_test_ordered, m_pred.ppgasp$lower95, m_pred.ppgasp$upper95)
lci(m_pred.ppgasp$lower95, m_pred.ppgasp$upper95)

## Vec-PPGP Fisher Scoring, With Nugget Estimate, Anistropic
start_time <- Sys.time()
temp = fisher(ini_param = c(eta,tau), tau = tau , nugget_est = TRUE, x = X_train_ordered,
              NNmatrix = nnmatrix_train, y = Y_train_ordered, kernel_type = "matern_3_2", alpha = 1.9,
              trend = matrix(1, nrow = 1000, 1), isotropic = F, zero_mean = "No")
end_time <- Sys.time()
end_time - start_time

learned_gamma = exp(-temp[1:length(temp)-1])
learned_nu = exp(temp[length(temp)])

list = predict(x = X_train_ordered, xp = X_test_ordered, NNmatrix = nnmatrix_all, 
               y = Y_train_ordered, gamma = learned_gamma, nu = learned_nu, 
               kernel_type = "matern_3_2", alpha = 1.9, q_95 = qnorm(0.975, 0, 1),
               trend = matrix(1,1000,1), testing_trend = matrix(1,500,1), 
               isotropic = F, zero_mean = "No")

mse(Y_test_ordered, list[[1]])
pci(Y_test_ordered, list[[2]], list[[3]])
lci(list[[2]], list[[3]])

################################################################################
## TITAN 2D Data
# Load and process the data
library(repmis)
source_data("https://github.com/MengyangGu/TITAN2D/blob/master/TITAN2D.rda?raw=True")

set.seed(2001)
maxmin_order <- GpGp::order_maxmin(input_variables[,1:3])
ordered_sample <- input_variables[maxmin_order,1:3]
ordered_response <- pyroclastic_flow_heights[maxmin_order,]

X_train_ordered = ordered_sample[1:50,]
X_test_ordered = ordered_sample[51:683,]
Y_train_ordered = ordered_response[1:50,] 
Y_test_ordered = ordered_response[51:683,]

n=dim(Y_train_ordered)[1]
n_testing=dim(Y_test_ordered)[1]

index_all_zero=NULL
for(i_loc in 1:dim(Y_train_ordered)[2]){
  if(sum(Y_train_ordered[,i_loc]==0)==50){
    index_all_zero=c(index_all_zero,i_loc)
  }
}
Y_train_ordered_log_1=log(Y_train_ordered+1)
k=dim(Y_train_ordered_log_1[,-index_all_zero])[2]
## Train PPGP 
start_time <- Sys.time()
m.ppgasp=ppgasp(design=X_train_ordered,
                response=as.matrix(Y_train_ordered_log_1[,-index_all_zero]),
                trend=cbind(rep(1,n),X_train_ordered[,1]),
                nugget.est=T,max_eval=100,num_initial_values=1,
                optimization='nelder-mead')
end_time <- Sys.time()
end_time - start_time

## PPGP prediction
start_time <- Sys.time()
pred_ppgasp=predict.ppgasp(m.ppgasp,X_test_ordered,
                           testing_trend=cbind(rep(1,n_testing),
                                              X_test_ordered[,1]))
end_time <- Sys.time()
end_time - start_time
# transform the output
m_pred_ppgasp_mean=exp(pred_ppgasp$mean)-1
m_pred_ppgasp_LB=exp(pred_ppgasp$lower95)-1
m_pred_ppgasp_UB=exp(pred_ppgasp$upper95)-1

m_pred_ppgasp_mean[which(m_pred_ppgasp_mean<0)]=0
m_pred_ppgasp_LB[which(m_pred_ppgasp_LB<0)]=0
m_pred_ppgasp_UB[which(m_pred_ppgasp_UB<0)]=0

mse(as.matrix(Y_test_ordered[,-index_all_zero]), m_pred_ppgasp_mean)
pci(as.matrix(Y_test_ordered[,-index_all_zero]), m_pred_ppgasp_LB, m_pred_ppgasp_UB)
lci(m_pred_ppgasp_LB, m_pred_ppgasp_UB)

## Train Vec-PPGP
set.seed(2001)
m = 50
nnmatrix_all <- GpGp::find_ordered_nn(ordered_sample, m = m)
nnmatrix_train <- GpGp::find_ordered_nn(X_train_ordered, m = m)

set.seed(2001)
gamma = runif(n = 3, min = gamma_range[1], max = gamma_range[2])
nu = runif(n = 1, min = nugget_range[1], max = nugget_range[2])
eta = log(1/gamma)
tau = log(nu)

set.seed(2001)
start_time = Sys.time()
res = optim(par = c(eta,tau), fn = neg_vecchia_marginal_log_likelihood, 
            gr = neg_vecchia_marginal_log_likelihood_deriv, method = "L-BFGS-B", 
            nu = 1, nugget_est = TRUE, x = as.matrix(X_train_ordered), 
            NNmatrix = nnmatrix_train, y = as.matrix(Y_train_ordered_log_1[,-index_all_zero]),
            kernel_type = "matern_5_2", alpha = 1.9, trend=cbind(rep(1,n),X_train_ordered[,1]),
            isotropic = F, zero_mean = "No", control = list(
              maxit = 10,
              factr = 1e7,  
              pgtol = 1e-6,
              trace = 0
            ))
end_time = Sys.time()
end_time - start_time

gamma_trained = exp(-res$par[-length(res$par)])
nu_trained = exp(res$par[length(res$par)])
start_time = Sys.time()
list = predict(x = as.matrix(X_train_ordered), xp = as.matrix(X_test_ordered), 
               NNmatrix = nnmatrix_all, y = as.matrix(Y_train_ordered_log_1[,-index_all_zero]), 
               gamma = gamma_trained, nu = nu_trained, kernel_type = "matern_5_2", 
               alpha = 1.9, q_95 = qnorm(0.975, 0, 1), trend = cbind(rep(1,n),X_train_ordered[,1]), 
               testing_trend = cbind(rep(1,n_testing),X_test_ordered[,1]), 
               isotropic = F,  zero_mean = "No")
end_time = Sys.time()
end_time - start_time

vec_ppgasp_mean=exp(list[[1]])-1
vec_ppgasp_LB=exp(list[[2]])-1
vec_ppgasp_UB=exp(list[[3]])-1
vec_ppgasp_mean[which(vec_ppgasp_mean<0)]=0
vec_ppgasp_LB[which(vec_ppgasp_LB<0)]=0
vec_ppgasp_UB[which(vec_ppgasp_UB<0)]=0

mse(as.matrix(Y_test_ordered[,-index_all_zero]), vec_ppgasp_mean)
pci(as.matrix(Y_test_ordered[,-index_all_zero]), vec_ppgasp_LB, vec_ppgasp_UB)
lci(vec_ppgasp_LB, vec_ppgasp_UB)

# Vec-PPGP: m = 25, Train Time Diff 14.860, Pred Time Diff 0.774, MSE 0.0927, coverage 0.942, length95 0.429
# Vec-PPGP: m = 50, Train Time Diff 14.808, Pred Time Diff 0.900, MSE 0.0864, coverage 0.942, length95 0.420
# PPGP: Train Time Diff 36.248, Pred Time Diff 0.302, MSE 0.0827, coverage 0.941, length95 0.412
# PPGP: Train Time Diff 3.120, Pred Time Diff 0.238, MSE 0.214, coverage 0.937, length95 0.642


################################################################################
## Effects of Different Orderings
time_mat = matrix(0, nrow = 4, ncol = 20)
mse_mat = matrix(0, nrow = 4, ncol = 20)
pci_mat = matrix(0, nrow = 4, ncol = 20)
lci_mat = matrix(0, nrow = 4, ncol = 20)

for (j in 1:20){
  temp = generate_data(n=1500, p =5, k=10, gamma_range = c(0.1, 2), sigma2_range = c(0.1, 0.5),nugget_range = c(0.1, 0.3))
  Xo = temp[[1]]
  Yo = temp[[2]]
  m = 50
  maxmin_order <- GpGp::order_maxmin(Xo)
  middleout_order <- GpGp::order_middleout(Xo)
  coor_sum_order <- GpGp::order_coordinate(Xo)
  random_order <- sample(seq(1500), 1500, replace = FALSE)
  
  time_list = c()
  mse_list = c()
  pci_list = c()
  lci_list = c()
  
  for (i in 1:4){
    order = list(maxmin_order, middleout_order, coor_sum_order, random_order)[i]
    ordered_sample <- Xo[order,]
    ordered_response <- Yo[order,]
    X_train_ordered <- ordered_sample[1:1000,]
    X_test_ordered <- ordered_sample[1001:1500,]
    Y_train_ordered <- ordered_response[1:1000,]
    Y_test_ordered <- ordered_response[1001:1500,]
    
    nnmatrix_all <- GpGp::find_ordered_nn(ordered_sample, m = m)
    nnmatrix_train <- GpGp::find_ordered_nn(X_train_ordered, m = m)
    gamma = runif(n = 5, min = 0.1, max = 2)
    nu = runif(n = 1, min = 0.1, max = 2)
    eta = log(1/gamma)
    tau = log(nu)
    
    start_time = Sys.time()
    res = optim(par = c(eta,tau), fn = neg_vecchia_marginal_log_likelihood, gr = neg_vecchia_marginal_log_likelihood_deriv, nu = 1, nugget_est = TRUE, x = X_train_ordered, NNmatrix = nnmatrix_train, kernel_type = "matern_3_2", y = Y_train_ordered, alpha = 1.9, method = "L-BFGS-B", trend = matrix(1,1000,1), isotropic = F, zero_mean = "Yes")
    end_time = Sys.time()
    time_list = c(time_list, as.numeric(end_time - start_time))
    
    gamma_trained = exp(-res$par[-length(res$par)])
    nu_trained = exp(res$par[length(res$par)])
    list = predict(x = X_train_ordered, xp = X_test_ordered, NNmatrix = nnmatrix_all, y = Y_train_ordered, gamma = gamma_trained, nu = nu_trained, kernel_type = "matern_3_2", trend = matrix(1,1000,1), testing_trend = matrix(1, 500, 1), isotropic = F, zero_mean = "Yes", alpha = 1.9, q = qnorm(0.975, 0, 1))
    
    mse_i = mse(Y_test_ordered, list[[1]])
    pci_i = pci(Y_test_ordered, list[[2]], list[[3]])
    lci_i = lci(list[[2]], list[[3]])
    mse_list = c(mse_list, mse_i)
    pci_list = c(pci_list, pci_i)
    lci_list = c(lci_list, lci_i)
  }
  time_mat[i,j] = time_list
  mse_mat[i,j] = mse_list
  pci_mat[i,j] = pci_list
  lci_mat[i,j] = lci_list
}

################################################################################
## Predict Group 4 rho
load("src/ini_4potentials_data.RData")

delete_index = which(rho_record[1,]==0)
sd_threshold = 0.01
alphalevel = 0.05
N = 2000
a = 1
L = 9
k = 1001

# n training
n = 1000
n_test = N - n
train_index = 1:n
group = 4

rho_group4 = rho_record[(1:N+(group-1)*N),]
be_V_group4 = be_V_ext_record[(1:N+(group-1)*N),]
be_mu_group4 = be_mu_record[(1:N+(group-1)*N)]
be_Omega_group4 = be_Omega_record[(1:N+(group-1)*N)]

input = matrix(be_mu_group4,ncol = k-length(delete_index),nrow = N)-be_V_group4[,-delete_index]
output = rho_group4[,-delete_index]

set.seed(2001)
maxmin_order <- GpGp::order_maxmin(input)
input_ordered <- input[maxmin_order,]
output_ordered <- output[maxmin_order, ]
be_Omega_ordered <- be_Omega_group4[maxmin_order]
be_Omega_test_ordered <- be_Omega_ordered[(n+1):N]

input_train_ordered = input_ordered[1:n,]
input_test_ordered = input_ordered[(n+1):N,]
output_train_ordered = output_ordered[1:n,] 
output_test_ordered = output_ordered[(n+1):N,]

## Train PPGP with nelder-mead, lbfgs wouldn't work
start_time = Sys.time()
m_GP=ppgasp(design=input_train_ordered,response=output_train_ordered,nugget.est=T,lower_bound=F,
            isotropic = T,optimization="nelder-mead",num_initial_values = 1)
end_time = Sys.time()
end_time - start_time
## Time diff is 54.102 s

## PPGP Prediction
m_pred=predict.ppgasp(m_GP,input_test_ordered)
pred_rho_di = m_pred$mean #predicted density with deleted index
sqrt(mse(pred_rho_di, output_test_ordered))
pci(output_test_ordered, m_pred$lower95, m_pred$upper95)
lci(m_pred$lower95, m_pred$upper95)

set.seed(2001)
m = 15
nnmatrix_all <- GpGp::find_ordered_nn(input_ordered, m = m)
nnmatrix_train <- GpGp::find_ordered_nn(input_train_ordered, m = m)

set.seed(2001)
gamma = runif(n = 1, min = gamma_range[1], max = gamma_range[2])
nu = runif(n = 1, min = nugget_range[1], max = nugget_range[2])
eta = log(1/gamma)
tau = log(nu)

set.seed(2001)
start_time = Sys.time()
res = optim(par = c(eta,tau), fn = neg_vecchia_marginal_log_likelihood, 
            gr = neg_vecchia_marginal_log_likelihood_deriv, 
            method = "L-BFGS-B", control = list(maxit = 10, factr = 1e5), 
            nu = 1, nugget_est = TRUE, x = as.matrix(input_train_ordered), 
            NNmatrix = nnmatrix_train, y = as.matrix(output_train_ordered),
            kernel_type = "matern_5_2", alpha = 1.9, 
            trend=cbind(rep(1,n),input_train_ordered[,1]), 
            isotropic = T, zero_mean = "Yes")
end_time = Sys.time()
end_time - start_time
## Time Diff is 15.668s

## Vec-PPGP Prediction
gamma_trained = exp(-res$par[-length(res$par)])
nu_trained = exp(res$par[length(res$par)])
start_time = Sys.time()
list = predict(x = as.matrix(input_train_ordered), xp = as.matrix(input_test_ordered), 
               NNmatrix = nnmatrix_all, y = as.matrix(output_train_ordered), 
               gamma = gamma_trained, nu = nu_trained, kernel_type = "matern_5_2", 
               alpha = 1.9, q_95 = qnorm(0.975, 0, 1), 
               trend = cbind(rep(1,n),input_train_ordered[,1]), 
               testing_trend = cbind(rep(1,n),input_test_ordered[,1]), 
               isotropic = T, zero_mean = "Yes")
end_time = Sys.time()
end_time - start_time

vec_mean = list[[1]]
vec_lb = list[[2]]
vec_ub = list[[3]]
sqrt(mse(vec_mean, output_test_ordered)) 
pci(output_test_ordered, vec_lb, vec_ub)
lci(vec_lb, vec_ub)


# To train other group, set group = 3, 2, or 1, and rerun the above code

# Group 4
# PPGP: Time difference of 52.281 secs, RMSE 0.033, coverage 0.975, length95 0.161
# Vec-PPGP: Time Diff 15.631, RMSE 0.029, coverage 0.953, length95 0.125

# Group 3
# PPGP: Time difference of 33.303 secs, RMSE 0.164, coverage 0.939, length95 0.643
# Vec-PPGP: Time Diff 11.713, RMSE 0.168, coverage 0.936, length95 0.670

# Group 2
# PPGP: Time difference of 53.125 secs, RMSE 1.62e-06, coverage 0.936, length95 6.67e-06
# Vec-PPGP: Time Diff 34.13, RMSE 4.54e-06, coverage 0.936, length95 9.481e-06

# Group 1
# PPGP: Time difference of 55.20 secs, RMSE 6.754e-07, coverage 0.954, length95 2.447e-06
# Vec-PPGP: Time Diff 17.570s, RMSE 7.641e-07, coverage 0.925, length95 2.594e-06

## Fisher-Scoring Prediction



# Group 1
# Fisher: Time difference of 8.379 secs, RMSE 1.071e-06, coverage 0.956, length95 4.876e-06

# Group 2
# Fisher: Time difference of 6.350 secs, RMSE 0.002, coverage 0.999, length95 0.016

# Group 3
# Fisher: Time difference of 6.403 secs, RMSE 0.088, coverage 0.982, length95 0.461

# Group 4
# Fisher: Time difference of 5.085 secs, RMSE 0.151, coverage 0.967, length95 0.320
