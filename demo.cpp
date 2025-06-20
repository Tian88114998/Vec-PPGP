
// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*- 
 
// we only include RcppEigen.h which pulls Rcpp.h in for us 
#include <Rcpp.h> 
#include <RcppEigen.h> 
#include <cmath> 
#include <chrono>
#include "ctools.h"
using namespace Rcpp;
using namespace std;
// [[Rcpp::depends(RcppEigen)]] 

// [[Rcpp::export]]
MatrixXd Exp2Sep(const MatrixXd &x1, const MatrixXd &x2, 
  const VectorXd gamma, const double nu, const double alpha) { 
  // the input matrix has n rows and p cols
  // the output is the correlation matrix
  int n1 = x1.rows();
  int n2 = x2.rows();
  int p = x1.cols();
  if (x1.cols() != x2.cols()){
    stop("dimension of x1 does not match with dimension of x2");
  }
  if (gamma.size() != p){
    stop("length of gamma does not match dimension of x");
  }
  MatrixXd covmat(n1, n2);
  VectorXd inv_gamma = gamma.array().inverse().pow(alpha);
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      double r = ((x1.row(i) - x2.row(j)).array().abs().pow(alpha) * inv_gamma.transpose().array()).sum();
      covmat(i, j) = exp(-r);
    }
  }
  if (n1 == n2){
    // when i element is only conditioned on one element we should not add nu. 
    if (n1 == 1 && x1 != x2){
      return covmat;
    }
    covmat += MatrixXd::Identity(n1, n1) * nu;
    return covmat;
  } else {
    return covmat; 
  }
}

// [[Rcpp::export]]
MatrixXd Matern_3_2_Sep(const MatrixXd &x1, const MatrixXd &x2, 
                 const VectorXd gamma, const double nu, const double alpha) { 
  // the input matrix has n rows and p cols
  // the output is the correlation matrix
  int n1 = x1.rows();
  int n2 = x2.rows();
  if (x1.cols() != x2.cols()){
    stop("dimension of x1 does not match with dimension of x2");
  }
  int p = x1.cols();
  if (gamma.size() != p){
    stop("length of gamma does not match dimension of x");
  }
  MatrixXd covmat(n1, n2);
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      double r = 1.0;
      for (int k = 0; k < p; k++){
        // change to squared value instead of absolute value
        double d = std::abs(x1(i, k) - x2(j, k));
        double temp = sqrt(3) * d / gamma(k);
        r *= (1+ temp) * exp(-temp);
      }
      covmat(i,j) = r;
    }
  }
  if (n1 == n2){
    // when i element is only conditioned on one element we should not add nu. 
    if (n1 == 1 && x1 != x2){
      return covmat;
    }
    covmat += MatrixXd::Identity(n1, n1) * nu;
    return covmat;
  } else {
    return covmat; 
  }
}

// [[Rcpp::export]]
MatrixXd Matern_5_2_Sep(const MatrixXd &x1, const MatrixXd &x2, 
                        const VectorXd gamma, const double nu, const double alpha) { 
  // the input matrix has n rows and p cols
  // the output is the correlation matrix
  int n1 = x1.rows();
  int n2 = x2.rows();
  if (x1.cols() != x2.cols()){
    stop("dimension of x1 does not match with dimension of x2");
  }
  int p = x1.cols();
  if (gamma.size() != p){
    stop("length of gamma does not match dimension of x");
  }
  MatrixXd covmat(n1, n2);
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      double r = 1.0;
      for (int k = 0; k < p; k++){
        // change to squared value instead of absolute value
        double d = std::abs(x1(i, k) - x2(j, k));
        double temp1 = sqrt(5) * d / gamma(k);
        double temp2 = pow(temp1, 2) / 3;
        r *= (1+ temp1 + temp2) * exp(-temp1);
      }
      covmat(i,j) = r;
    }
  }
  if (n1 == n2){
    // when i element is only conditioned on one element we should not add nu. 
    if (n1 == 1 && x1 != x2){
      return covmat;
    }
    covmat += MatrixXd::Identity(n1, n1) * nu;
    return covmat;
  } else {
    return covmat; 
  }
}

// [[Rcpp::export]]
MatrixXd Exp2Sep_deriv_gamma_k(const MatrixXd &x1, const MatrixXd &x2, 
               const VectorXd gamma, const double nu, 
               const double alpha, const int k){
  // the output is the derivative of the correlation matrix 
  // with respect to the kth of gamma
  int n1 = x1.rows();
  int n2 = x2.rows();
  if (x1.cols() != x2.cols()){
    stop("dimension of x1 does not match with dimension of x2");
  }
  if (gamma.size() != x1.cols()){
    stop("length of gamma does not match dimension of x");
  }
  MatrixXd res(n1,n2);
  // first construct the correlation matrix
  MatrixXd covmat = Exp2Sep(x1, x2, gamma, nu, alpha);
  // for each gamma k, calculate the matrix derivative
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      double d = abs(x1(i,k)-x2(j,k));
      double extra = alpha * 
        pow(gamma(k), - alpha - 1) * pow(d, alpha);
       res(i, j) = covmat(i,j) * extra;
     }
  }
  return res;
}

// [[Rcpp::export]]
MatrixXd Matern_3_2_Sep_deriv_gamma_k(const MatrixXd &x1, const MatrixXd &x2, 
                               const VectorXd gamma, const double nu, 
                               const double alpha, const int k){
  // the output is the derivative of the correlation matrix 
  // with respect to the kth of gamma
  int n1 = x1.rows();
  int n2 = x2.rows();
  if (x1.cols() != x2.cols()){
    stop("dimension of x1 does not match with dimension of x2");
  }
  if (gamma.size() != x1.cols()){
    stop("length of gamma does not match dimension of x");
  }
  MatrixXd res(n1,n2);
  // first construct the correlation matrix
  MatrixXd covmat = Matern_3_2_Sep(x1, x2, gamma, nu, alpha);
  // for each gamma k, calculate the matrix derivative
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      double d = abs(x1(i,k)-x2(j,k));
      double temp1 = 1 + sqrt(3) * d / gamma(k);
      double temp2 = 3 * pow(d, 2) / pow(gamma(k), 3);
      res(i, j) = covmat(i,j) * temp2 / temp1;
    }
  }
  return res;
}

// [[Rcpp::export]]
MatrixXd Matern_5_2_Sep_deriv_gamma_k(const MatrixXd &x1, const MatrixXd &x2, 
                                      const VectorXd gamma, const double nu, 
                                      const double alpha, const int k){
  // the output is the derivative of the correlation matrix 
  // with respect to the kth of gamma
  int n1 = x1.rows();
  int n2 = x2.rows();
  if (x1.cols() != x2.cols()){
    stop("dimension of x1 does not match with dimension of x2");
  }
  if (gamma.size() != x1.cols()){
    stop("length of gamma does not match dimension of x");
  }
  MatrixXd res(n1,n2);
  // first construct the correlation matrix
  MatrixXd covmat = Matern_5_2_Sep(x1, x2, gamma, nu, alpha);
  // for each gamma k, calculate the matrix derivative
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      double d = abs(x1(i,k)-x2(j,k));
      double temp1 = 1 + sqrt(5) * d / gamma(k) + 5 * pow(d,2)/(3*pow(gamma(k), 2));
      double temp2 = 5 * sqrt(5) * pow(d, 3) / (3 * pow(gamma(k), 4)) + 5 * pow(d, 2)/ (3 * pow(gamma(k), 3));
      res(i, j) = covmat(i,j) * temp2 / temp1;
    }
  }
  return res;
}

// [[Rcpp::export]]
MatrixXd R_inv_y(const MatrixXd &R, const VectorXd &y){
  
  LLT<MatrixXd> lltOfR(R);             // compute the cholesky decomposition of R called lltofR
  MatrixXd L = lltOfR.matrixL();   //retrieve factor L  in the decomposition
  MatrixXd res=L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(y)); //forward and backward substitution
  
  return res;
}

// [[Rcpp::export]]
// Given a vector of the neighbor indices, return the ith column of the U
Rcpp::List Uentries(const MatrixXd &x, const VectorXd &idx, const int i, const VectorXd gamma, 
             const double nu, const string kernel_type, const double alpha) {
  //x is the ordered input matrix, idx is a vector contains all neighbor indices of ith element
  //i indicates the ith element 
  
  // this function work when the idx set is not all NA. 
  int m = idx.size();
  MatrixXd selected_neighbors(m, x.cols());
  // select the neighbors of i according to the idx set
  for (int j = 0; j < m; ++j){
  // an index change happened here
    selected_neighbors.row(j) = x.row(idx[j]-1);
  }
  // select the ith element 
  MatrixXd i_element = x.row(i);
  // create a place holder for the Uentries
  VectorXd i_column(m+1);
  
  std::function<MatrixXd(const MatrixXd&, const MatrixXd&, const VectorXd&, double, double)> kernel_func;
  
  if (kernel_type == "pow_exp"){
    kernel_func = Exp2Sep;
  } else if (kernel_type == "matern_3_2"){
    kernel_func = Matern_3_2_Sep;
  } else if (kernel_type == "matern_5_2"){
    kernel_func = Matern_5_2_Sep;
  }
  MatrixXd Sigma_cc = kernel_func(selected_neighbors,selected_neighbors, 
                     gamma, nu, alpha);
  MatrixXd Sigma_ci = kernel_func(selected_neighbors, i_element, gamma, nu,
                     alpha);
  MatrixXd B_i = R_inv_y(Sigma_cc, Sigma_ci).transpose();
  MatrixXd D_i = kernel_func(i_element, i_element, gamma, nu,
                   alpha) - B_i * Sigma_ci;
  // D_i_half_inv is the diagonal term
  double D_i_half_inv = 1/sqrt(D_i(0,0));
  i_column.head(m) = -D_i_half_inv * B_i;
  i_column(m) = D_i_half_inv;
  return Rcpp::List::create(i_column, D_i_half_inv);
}

// [[Rcpp::export]]
// Given the idx vector and the corresponding U entries, return the Ucolumn
VectorXd Ucolumn(const VectorXd &i_column, const VectorXd &idx, const int &i,
                 const int &n) {
  VectorXd Ucol = VectorXd::Zero(n);
  // set the diagonal term
  Ucol(i) = i_column(i_column.size()-1);
  // set the rest entries corresponding to the idx
  for (int j = 0; j < idx.size(); ++j){
    double entry = i_column(j);
    int Uidx = idx(j) - 1;
    Ucol(Uidx) = entry;
  }
  return Ucol;
}

// [[Rcpp::export]]
Rcpp::List Umatrix(const MatrixXd &x, const MatrixXd &NNmatrix, 
                 const VectorXd gamma, const double nu, const string kernel_type,
                     const double alpha) {
  int n = x.rows();
  // the first column of the NNmatrix is not useful
  int m = NNmatrix.cols() - 1;
  MatrixXd U = MatrixXd::Zero(n,n);
  // return the sum of log diagonal terms for log-likelihood
  double sum_log_diag_U = 0;
  // set the first diagonal term since it has no neighbor
  double diag_11;
  if (kernel_type == "pow_exp"){
     diag_11 = 1/sqrt(Exp2Sep(x.row(0),x.row(0), gamma, nu, alpha)(0,0));
  } else if (kernel_type == "matern_3_2"){
     diag_11 = 1/sqrt(Matern_3_2_Sep(x.row(0),x.row(0), gamma, nu, alpha)(0,0));
  } else if (kernel_type == "matern_5_2"){
     diag_11 = 1/sqrt(Matern_5_2_Sep(x.row(0),x.row(0), gamma, nu, alpha)(0,0));
  } else {
    Rcpp::stop("Unknown kernel type.");
  }
  U(0,0) = diag_11;
  sum_log_diag_U += log(diag_11);
  for (int i = 1; i < n; ++i){
    // for the first m elements, there are NA in the NNmatrix. Select only the non-NA's.
    // define the vector before the if-else
    VectorXd idx;
    if (i < m) {
      idx = NNmatrix.block(i, 1, 1, i).transpose();
      List temp = Uentries(x, idx, i, gamma, nu, kernel_type, alpha);
      VectorXd icolumn = temp[0];
      double diag = temp[1];
      VectorXd ucol = Ucolumn(icolumn, idx, i, n);
      U.col(i) = ucol;
      sum_log_diag_U += log(diag);
    } else {
      idx = NNmatrix.row(i).segment(1, m);
      List temp = Uentries(x, idx, i, gamma, nu, kernel_type, alpha);
      VectorXd icolumn = temp[0];
      double diag = temp[1];
      VectorXd ucol = Ucolumn(icolumn, idx, i, n);
      U.col(i) = ucol;
      sum_log_diag_U += log(diag);
    }
  }
  return Rcpp::List::create(U, sum_log_diag_U);
}

using KernelFunc = std::function<MatrixXd(const MatrixXd&, const MatrixXd&, const VectorXd&, double, double)>;
using KernelFunc_deriv = std::function<MatrixXd(const MatrixXd&, const MatrixXd&, const VectorXd&, double, double, int)>;

// [[Rcpp::export]]
Rcpp::List Umatrix1(const MatrixXd &x, const MatrixXd &NNmatrix, 
                    const VectorXd gamma, const double nu, const string kernel_type,
                    const double alpha) {
  int n = x.rows();
  int m = NNmatrix.cols() - 1;
  MatrixXd U = MatrixXd::Zero(n, n);
  double sum_log_diag_U = 0;
  
  // Select kernel function once
  KernelFunc kernel_func;
  if (kernel_type == "pow_exp") {
    kernel_func = Exp2Sep;
  } else if (kernel_type == "matern_3_2") {
    kernel_func = Matern_3_2_Sep;
  } else if (kernel_type == "matern_5_2") {
    kernel_func = Matern_5_2_Sep;
  } else {
    Rcpp::stop("Unknown kernel type.");
  }
  
  // Precompute diag_11 once
  double diag_11 = 1.0 / sqrt(kernel_func(x.row(0), x.row(0), gamma, nu, alpha)(0,0));
  U(0, 0) = diag_11;
  sum_log_diag_U += std::log(diag_11);
  
  for (int i = 1; i < n; ++i) {
    int num_neighbors = std::min(i, m);
    VectorXd idx = NNmatrix.block(i, 1, 1, num_neighbors).transpose();
    
    MatrixXd selected_neighbors(num_neighbors, x.cols());
    for (int j = 0; j < num_neighbors; ++j){
      selected_neighbors.row(j) = x.row(idx[j] - 1);
    }
    
    MatrixXd i_element = x.row(i);
    MatrixXd Sigma_cc = kernel_func(selected_neighbors, selected_neighbors, gamma, nu, alpha);
    MatrixXd Sigma_ci = kernel_func(selected_neighbors, i_element, gamma, nu, alpha);
    MatrixXd B_i = R_inv_y(Sigma_cc, Sigma_ci).transpose();
    MatrixXd D_i = kernel_func(i_element, i_element, gamma, nu, alpha) - B_i * Sigma_ci;
    double D_i_half_inv = 1.0 / sqrt(D_i(0, 0));
    
    VectorXd i_column(num_neighbors + 1);
    i_column.head(num_neighbors) = -D_i_half_inv * B_i;
    i_column(num_neighbors) = D_i_half_inv;
    
    // Fast Ucol insert
    U(i, i) = D_i_half_inv;
    for (int j = 0; j < num_neighbors; ++j) {
      U(static_cast<int>(idx[j]) - 1, i) = i_column[j];
    }
    
    sum_log_diag_U += std::log(D_i_half_inv);
  }
  return Rcpp::List::create(U, sum_log_diag_U);
}

//[[Rcpp::export]]
double neg_vecchia_marginal_log_likelihood(const VectorXd params, double nu, const bool nugget_est, 
                                           const MatrixXd &x, const MatrixXd &NNmatrix, const MatrixXd &y,
                                           const String kernel_type, const double alpha, 
                                           const MatrixXd& trend, const String zero_mean = "Yes"){
  // y is a nxk matrix
  int k = y.cols();
  int n = y.rows();
  VectorXd gamma;
  if (nugget_est){
    gamma = params.head(params.size()-1);
    nu = params(params.size()-1);
  } else{
    gamma = params;
  }
  List temp = Umatrix(x, NNmatrix, gamma, nu, kernel_type, alpha);
  MatrixXd U = temp[0];
  MatrixXd UUt = U * U.transpose();
  double res = 0.0;
  double sum_log_diag_U = temp[1];
  if (zero_mean == "Yes"){
    for (int i = 0; i < k; ++i){
      res += log(y.col(i).transpose() * UUt * y.col(i));
    }
    res *= -n / 2.0;
    res += k * sum_log_diag_U;
  } else{
    int q = trend.cols();
    MatrixXd HtUUt = trend.transpose() * UUt;
    MatrixXd N = HtUUt * trend;
    LLT<MatrixXd> llt(N);
    MatrixXd N_inv_H_t_UU_t = llt.solve(HtUUt);
    MatrixXd L = llt.matrixL();
    double N_log_det = L.diagonal().array().log().matrix().sum();
    for (int j = 0; j < k; ++j) {
      MatrixXd y_j = y.col(j);
      MatrixXd temp = trend * (N_inv_H_t_UU_t * y_j);
      MatrixXd Qy_j = UUt * (y_j - temp);
      res += log((y_j.transpose() * Qy_j)(0,0));
    }
    // delete by 2.0 but not 2
    res *= -(n-q)/2.0;
    res += k * sum_log_diag_U;
    res -= k * N_log_det;
  }
  Rcpp::Rcout << "params" << std::endl;
  Rcpp::Rcout << params << std::endl;
  return -res;
}

// TODO: check if this function is correct. 
// [[Rcpp::export]]
VectorXd neg_vecchia_marginal_log_likelihood_deriv(const VectorXd params, double nu, const bool nugget_est,
                                                   const MatrixXd &x, const MatrixXd &NNmatrix,
                                                   const MatrixXd &y,
                                                   const String kernel_type, const double alpha, 
                                                   const MatrixXd &trend, const String zero_mean = "Yes"){
  // if nugget_est = TRUE, params should contain gamma and nu, and use the nu in params
  // if nuggest_est = FALSE, params should contain only gamma, and use the nu as given.
  // y is n * k 
  using namespace std::chrono;
  
  int k = y.cols();
  int n = y.rows();
  int m = NNmatrix.cols() - 1;
  MatrixXd H = trend;
  MatrixXd U = MatrixXd::Zero(n,n);
  KernelFunc kernel_func;
  KernelFunc_deriv kernel_func_deriv_gamma_k;
  if (kernel_type == "pow_exp"){
    kernel_func = Exp2Sep;
    kernel_func_deriv_gamma_k = Exp2Sep_deriv_gamma_k;
  } else if (kernel_type == "matern_3_2"){
    kernel_func = Matern_3_2_Sep;
    kernel_func_deriv_gamma_k = Matern_3_2_Sep_deriv_gamma_k;
  } else if (kernel_type == "matern_5_2"){
    kernel_func = Matern_5_2_Sep;
    kernel_func_deriv_gamma_k = Matern_5_2_Sep_deriv_gamma_k;
  } else {
    Rcpp::stop("Unknown kernel type.");
  }
  if (nugget_est){
    auto start = high_resolution_clock::now();
    int p = params.size()-1;
    VectorXd gamma = params.head(p);
    nu = params(p);
    U(0,0) = 1/sqrt(kernel_func(x.row(0),x.row(0), gamma, nu, alpha)(0,0));
  
    VectorXd grad = VectorXd::Zero(p+1);
    VectorXd sum_log_deriv = VectorXd::Zero(p+1);
    
    // a list of matrix to store dU/dgamma_k and dU/dnu 
    std::vector<MatrixXd> matrix_list(p+1, MatrixXd::Zero(n, n));
    // first set the (1,1) element of dU/dnu
    matrix_list[p](0,0) = -0.5 * pow(kernel_func(x.row(0), x.row(0), gamma, nu, alpha)(0,0), -1.5);
    // the (1,1) element of du/dgamma is 0
    
    sum_log_deriv(p) -= 0.5 * 1/kernel_func(x.row(0), x.row(0), gamma, nu, alpha)(0,0);
    // deal with i = 0 case separately since there is no neighbor
    for (int i = 1; i < n; ++i){
      int num_neighbors = std::min(i,m);
      VectorXd idx = NNmatrix.block(i, 1, 1, num_neighbors).transpose();
      MatrixXd selected_neighbors(num_neighbors, x.cols());
      for (int j = 0; j < num_neighbors; ++j){
        selected_neighbors.row(j) = x.row(idx[j] - 1);
      }
      
      // first calculate the gradient of gamma
      MatrixXd i_element = x.row(i);
      MatrixXd Sigma_cc = kernel_func(selected_neighbors, selected_neighbors, gamma, nu, alpha);
      MatrixXd Sigma_ci = kernel_func(selected_neighbors, i_element, gamma, nu, alpha);
      MatrixXd B_i_t = R_inv_y(Sigma_cc, Sigma_ci);
      MatrixXd B_i = B_i_t.transpose();
      MatrixXd D_i = kernel_func(i_element, i_element, gamma, nu,
                             alpha) - B_i * Sigma_ci;
      // this is 1/sigma
      double D_i_half_inv = 1/sqrt(D_i(0,0));
      double D_i_half_inv2 = D_i_half_inv * D_i_half_inv;
      double D_i_half_inv3 = D_i_half_inv2 * D_i_half_inv;
      
      U(i, i) = D_i_half_inv;
      
      // construct the dU matrix
      VectorXd i_column(num_neighbors + 1);
      i_column.head(num_neighbors) = -D_i_half_inv * B_i;
      i_column(num_neighbors) = D_i_half_inv;
      for (int j = 0; j < num_neighbors; ++j) {
        U(static_cast<int>(idx[j]) - 1, i) = i_column[j];
      }
      for (int j = 0; j < p; ++j){
        MatrixXd Sigma_ci_deriv_gamma_j = kernel_func_deriv_gamma_k(selected_neighbors, i_element, gamma, nu, alpha, j);
        MatrixXd Sigma_cc_deriv_gamma_j = kernel_func_deriv_gamma_k(selected_neighbors, selected_neighbors, gamma, nu, alpha, j);
        double Sigma_ci_deriv_B_i_t = (Sigma_ci_deriv_gamma_j.transpose() * B_i_t)(0,0);
        double A_j = 2 * Sigma_ci_deriv_B_i_t - (B_i * (Sigma_cc_deriv_gamma_j * B_i_t))(0,0);
        // this finishes the first part
        sum_log_deriv(j) += 0.5 * D_i_half_inv2 * A_j;
        // the second part we need to dU/dgamma
        double diag_j = 0.5 * D_i_half_inv3 * A_j;
        // see the notes for these computations
        MatrixXd Sigma_ic_deriv_Sigma_cc_inv = R_inv_y(Sigma_cc, Sigma_ci_deriv_gamma_j).transpose();
        MatrixXd temp2 = B_i * Sigma_cc_deriv_gamma_j;
        MatrixXd Sigma_ic_Sigma_cc_inv_deriv = R_inv_y(Sigma_cc, temp2.transpose()).transpose();
        MatrixXd B_i_dgamma = Sigma_ic_deriv_Sigma_cc_inv - Sigma_ic_Sigma_cc_inv_deriv;
        MatrixXd col_j = - diag_j * B_i - D_i_half_inv * B_i_dgamma;
        // get the i_column of dU/dgamma
        VectorXd i_column_j(col_j.cols() + 1);
        i_column_j.head(col_j.cols()) = col_j;
        i_column_j(col_j.cols()) = diag_j;
        // use Ucolumn function to construct the column of dU/dgamma
        VectorXd ucol = Ucolumn(i_column_j, idx, i, n);
        // update the matrix's column
        matrix_list[j].col(i) = ucol;
        
        // the top left term of dU/dgamma is 0 so we don't need to calculate it separately
      }
      
      double B_i_squared = B_i.row(0).squaredNorm();
      double W = 1.0 + B_i_squared;
      
      sum_log_deriv(p) -= 0.5 * D_i_half_inv2 * W;
      double diag_nu = -0.5 * D_i_half_inv3 * W;
      MatrixXd off_diag_nu = -diag_nu * B_i + D_i_half_inv * R_inv_y(Sigma_cc, B_i.transpose()).transpose();
      // get the i_column of dU/dnu
      i_column(off_diag_nu.cols() + 1);
      i_column.head(off_diag_nu.cols()) = off_diag_nu;
      i_column(off_diag_nu.cols()) = diag_nu;
      VectorXd ucol = Ucolumn(i_column, idx, i, n);
      matrix_list[p].col(i) = ucol;
    }
    if (zero_mean == "Yes"){
      MatrixXd Ut = U.transpose();
      for (int j = 0; j < p+1; ++j){
        grad(j) -= k * sum_log_deriv(j);
      }
      for (int z = 0; z < k; ++z){
        VectorXd y_z = y.col(z);
        VectorXd Ut_y_z = Ut * y_z;
        VectorXd UUt_y_z = U * Ut_y_z;
        double temp3 =  y_z.dot(UUt_y_z);
        for (int j = 0; j < p+1; ++j){
          double temp4 = y_z.dot(matrix_list[j] * Ut_y_z);
          grad(j) += n * 1/temp3 * temp4;
        }
      }
    } else{
      int q = H.cols();
      MatrixXd Ut = U.transpose();
      MatrixXd UtH = Ut*H;
      MatrixXd UUtH = U * UtH;
      MatrixXd Ht = H.transpose();
      MatrixXd N = Ht * UUtH;
      LLT<MatrixXd> llt(N);
      MatrixXd N_inv = llt.solve(MatrixXd::Identity(q, q));
      MatrixXd N_inv_Ht = N_inv * Ht;
      MatrixXd H_N_inv_Ht = H * N_inv * Ht;
      
      for (int j = 0; j < p+1; ++j){
        grad(j) -= k * sum_log_deriv(j);
        MatrixXd MH = matrix_list[j] * UtH;
        MatrixXd HtMH = Ht * MH;
        grad(j) += k * (N_inv * HtMH).diagonal().sum();
      }
      for (int z = 0; z < k; ++z){
        VectorXd y_z = y.col(z);
        MatrixXd Ut_y_z = Ut * y_z;
        MatrixXd UUt_y_z = U * Ut_y_z;
        // Q1 = Ut_H_N_inv_Ht_UUt_y_z
        MatrixXd Q1 = Ut * (H_N_inv_Ht * UUt_y_z);
        VectorXd Qy_z = UUt_y_z - U * Q1;
        double temp5 = y_z.dot(Qy_z);
        for (int j = 0; j < p+1; ++j){
          MatrixXd first_term = matrix_list[j] * Ut_y_z;
          MatrixXd second_term = matrix_list[j] * Q1;
          MatrixXd third_term = U * (Ut * (H_N_inv_Ht * second_term));
          MatrixXd fourth_term = U * (Ut * (H_N_inv_Ht * first_term));
          VectorXd Qj_y_z = first_term - second_term + third_term - fourth_term;
          double temp6 = y_z.dot(Qj_y_z);
          grad(j) += (n-q) * 1/temp5 * temp6;
        }
      }
    }
    // clip the gradient
    VectorXd grad_clip(p+1);
    for (int j = 0; j < p+1; ++j){
      if (std::abs(grad(j)) > 1e5) {
        grad_clip(j) = std::copysign(1e5, grad(j));
      } else{
        grad_clip(j) = grad(j);
      }
    }
    Rcpp::Rcout << "grad_clip" << std::endl;
    Rcpp::Rcout << grad_clip << std::endl;
    auto end = high_resolution_clock::now();
    duration<double> diff = end - start;
    Rcpp::Rcout << "Deriv with nugget section took " << diff.count() << " seconds.\n";
    return grad_clip;
  } else{
    auto start = high_resolution_clock::now();
    int p = params.size();
    VectorXd gamma = params;
    U(0,0) = 1/sqrt(kernel_func(x.row(0),x.row(0), gamma, nu, alpha)(0,0));
    
    VectorXd grad = VectorXd::Zero(p);
    VectorXd sum_log_deriv = VectorXd::Zero(p);
    std::vector<MatrixXd> matrix_list(p, MatrixXd::Zero(n, n));
    
    // deal with i = 0 case separately since there is no neighbor
    for (int i = 1; i < n; ++i){
      int num_neighbors = std::min(i,m);
      VectorXd idx = NNmatrix.block(i, 1, 1, num_neighbors).transpose();
      MatrixXd selected_neighbors(num_neighbors, x.cols());
      for (int j = 0; j < num_neighbors; ++j){
        selected_neighbors.row(j) = x.row(idx[j] - 1);
      }
      
      // first calculate the gradient of gamma
      MatrixXd i_element = x.row(i);
      MatrixXd Sigma_cc = kernel_func(selected_neighbors, selected_neighbors, gamma, nu, alpha);
      MatrixXd Sigma_ci = kernel_func(selected_neighbors, i_element, gamma, nu, alpha);
      MatrixXd B_i_t = R_inv_y(Sigma_cc, Sigma_ci);
      MatrixXd B_i = B_i_t.transpose();
      MatrixXd D_i = kernel_func(i_element, i_element, gamma, nu,
                             alpha) - B_i * Sigma_ci;
      // this is 1/sigma
      double D_i_half_inv = 1/sqrt(D_i(0,0));
      double D_i_half_inv2 = D_i_half_inv * D_i_half_inv;
      double D_i_half_inv3 = D_i_half_inv2 * D_i_half_inv;
      
      U(i, i) = D_i_half_inv;
      
      // construct the U matrix
      VectorXd i_column(num_neighbors + 1);
      i_column.head(num_neighbors) = -D_i_half_inv * B_i;
      i_column(num_neighbors) = D_i_half_inv;
      for (int j = 0; j < num_neighbors; ++j) {
        U(static_cast<int>(idx[j]) - 1, i) = i_column[j];
      }
      
      for (int j = 0; j < p; ++j){
        MatrixXd Sigma_ci_deriv_gamma_j = kernel_func_deriv_gamma_k(selected_neighbors, i_element, gamma, nu, alpha, j);
        MatrixXd Sigma_cc_deriv_gamma_j = kernel_func_deriv_gamma_k(selected_neighbors, selected_neighbors, gamma, nu, alpha, j);
        double Sigma_ci_deriv_B_i_t = (Sigma_ci_deriv_gamma_j.transpose() * B_i_t)(0,0);
        double A_j = 2* Sigma_ci_deriv_B_i_t - (B_i * Sigma_cc_deriv_gamma_j * B_i_t)(0,0);
        // this finishes the first part
        sum_log_deriv(j) += 0.5 * D_i_half_inv2 * A_j;
        // the second part we need to dU/dgamma
        double diag_j = 0.5 * D_i_half_inv3 * A_j;
        // see the notes for these computations
        MatrixXd Sigma_ic_deriv_Sigma_cc_inv = R_inv_y(Sigma_cc, Sigma_ci_deriv_gamma_j).transpose();
        MatrixXd temp2 = B_i * Sigma_cc_deriv_gamma_j;
        MatrixXd Sigma_ic_Sigma_cc_inv_deriv = R_inv_y(Sigma_cc, temp2.transpose()).transpose();
        MatrixXd B_i_dgamma = Sigma_ic_deriv_Sigma_cc_inv - Sigma_ic_Sigma_cc_inv_deriv;
        MatrixXd col_j = - diag_j * B_i - D_i_half_inv * B_i_dgamma;
        // get the i_column of dU/dgamma
        VectorXd i_column_j(col_j.cols() + 1);
        i_column_j.head(col_j.cols()) = col_j;
        i_column_j(col_j.cols()) = diag_j;
        // use Ucolumn function to construct the column of dU/dgamma
        VectorXd ucol = Ucolumn(i_column_j, idx, i, n);
        // update the matrix's column
        matrix_list[j].col(i) = ucol;
        // the top left term of dU/dgamma is 0 so we don't need to calculate it separately
      }
    }
    // finish the gradient of gamma
    if (zero_mean == "Yes"){
      MatrixXd Ut = U.transpose();
      for (int j = 0; j < p; ++j){
        grad(j) -= k * sum_log_deriv(j);
      }
      for (int z = 0; z < k; ++z){
        VectorXd y_z = y.col(z);
        VectorXd Ut_y_z = Ut * y_z;
        VectorXd UUt_y_z = U * Ut_y_z;
        double temp3 =  y_z.dot(UUt_y_z);
        for (int j = 0; j < p; ++j){
          double temp4 = y_z.dot(matrix_list[j] * Ut_y_z);
          grad(j) += n * 1/temp3 * temp4;
        }
      }
    } else{
      MatrixXd Ut = U.transpose();
      MatrixXd UUt = U*Ut;
      int q = H.cols();
      MatrixXd Ht = H.transpose();
      MatrixXd N = Ht * UUt * H;
      LLT<MatrixXd> llt(N);
      MatrixXd N_inv = llt.solve(MatrixXd::Identity(q, q));
      MatrixXd H_N_inv_Ht = H * N_inv * Ht;
      MatrixXd Q1 = MatrixXd::Identity(n,n) - H_N_inv_Ht * UUt;
      MatrixXd Q = UUt * Q1;
      MatrixXd H_N_Ht = H * N * Ht;
      for (int j = 0; j < p; ++j){
        MatrixXd M = matrix_list[j] * Ut;
        MatrixXd dQ = M * Q1 + UUt * (H_N_inv_Ht * M * H_N_inv_Ht * UUt - H_N_inv_Ht * M);
        grad(j) -= k * sum_log_deriv(j);
        grad(j) += k * (N_inv * Ht * M * H).diagonal().sum();
        for (int z = 0; z < k; ++z) {
          MatrixXd temp6 = y.col(z).transpose() * Q * y.col(z);
          MatrixXd temp7 = y.col(z).transpose() * dQ * y.col(z);
          grad(j) += (n-q) * 1/temp6(0,0) * temp7(0,0);
        }
      }
    }
    // clip the gradient
    VectorXd grad_clip(p);
    for (int j = 0; j < p; ++j){
      if (std::abs(grad(j)) > 1e5) {
        grad_clip(j) = std::copysign(1e5, grad(j));
      } else{
        grad_clip(j) = grad(j);
      }
    }
    Rcpp::Rcout << "grad_clip" << std::endl;
    Rcpp::Rcout << grad_clip << std::endl;
    auto end = high_resolution_clock::now();
    duration<double> diff = end - start;
    Rcpp::Rcout << "Deriv w/o nugget took " << diff.count() << " seconds.\n";
    return grad_clip;
  }
}

// [[Rcpp::export]]
// make prediction on new points 
List predict(const MatrixXd &x, const MatrixXd &xp, const MatrixXd &NNmatrix,
               const MatrixXd &y, const VectorXd gamma, const double nu, 
               const String kernel_type, const double alpha, const double q_95, 
               const MatrixXd& trend, 
               const MatrixXd& testing_trend,
               const String zero_mean = "Yes")
{
  int k = y.cols();
  int n = y.rows();
  int np = xp.rows();
  MatrixXd allx(x.rows() + xp.rows(), x.cols());
  allx.topRows(x.rows()) = x;
  allx.bottomRows(xp.rows()) = xp;
  // create the U matrix and partitions
  MatrixXd U = Umatrix(allx, NNmatrix, gamma, nu, kernel_type, alpha)[0];
  MatrixXd UoUot = U.block(0,0,n,n) * U.block(0,0,n,n).transpose() ;
  MatrixXd Up = U.block(n,n,np,np);
  MatrixXd Uopt = U.block(0, n, n, np).transpose();

  MatrixXd pred_mean(np, k);
  MatrixXd upper95(np, k);
  MatrixXd lower95(np, k);
  VectorXd Sigma_ii(np);
  for (int i = 0; i < np; i++){
    VectorXd i_vec = VectorXd::Zero(np);
    i_vec[i] = 1;
    VectorXd temp = Up.triangularView<Eigen::Upper>().solve(i_vec);
    Sigma_ii[i] = temp.transpose() * temp;
    // compute Sigma_ii
  }
  if (zero_mean == "Yes"){
    for (int j = 0; j < k; j++){
      MatrixXd y_j = y.col(j);
      double sigma2_j = (y_j.transpose() * UoUot * y_j)(0,0)/n;
      MatrixXd temp_mean = - Up.transpose().triangularView<Eigen::Lower>().solve(Uopt * y_j);
      pred_mean.col(j) = temp_mean;
      MatrixXd temp_sd = q_95 * ((sigma2_j * Sigma_ii.array())).sqrt().matrix();
      MatrixXd temp_bound = temp_mean + temp_sd;
      upper95.col(j) = temp_bound;
      temp_bound = temp_mean - temp_sd;
      lower95.col(j) = temp_bound;
    }
  } else{
    MatrixXd H = trend; 
    MatrixXd Hp = testing_trend;
    int q = H.cols();
    for (int j = 0; j < k; j++){
      MatrixXd y_j = y.col(j);
      MatrixXd temp = H.transpose() * UoUot * H;
      LLT<MatrixXd> llt(temp);
      MatrixXd temp_inv = llt.solve(MatrixXd::Identity(q, q));
      MatrixXd theta_j = temp_inv * H.transpose() * (UoUot * y_j);
      MatrixXd temp_mean = Hp * theta_j - Up.transpose().triangularView<Eigen::Lower>().solve(Uopt * (y_j - H * theta_j));
      pred_mean.col(j) = temp_mean;
      double sigma2_j = ((y_j- H * theta_j).transpose() * UoUot * (y_j- H* theta_j))(0,0)/(n-q);
      MatrixXd temp_sd = q_95 * ((sigma2_j * Sigma_ii.array())).sqrt().matrix();
      upper95.col(j) = temp_mean + temp_sd;
      lower95.col(j) = temp_mean - temp_sd;
    }
  }
  return Rcpp::List::create(pred_mean, lower95, upper95);
}

// [[Rcpp::export]]
List fisher_scoring(const VectorXd params, double nu, const bool nugget_est,
                    const MatrixXd &x, const MatrixXd &NNmatrix,
                    const MatrixXd &y,
                    const String kernel_type, const double alpha, 
                    const MatrixXd &trend, const String zero_mean = "Yes"){
  using namespace std::chrono;
  auto start = high_resolution_clock::now();
  int k = y.cols();
  int n = y.rows();
  int m = NNmatrix.cols() - 1;
  MatrixXd H = trend;
  KernelFunc kernel_func;
  KernelFunc_deriv kernel_func_deriv_gamma_k;
  if (kernel_type == "pow_exp"){
    kernel_func = Exp2Sep;
    kernel_func_deriv_gamma_k = Exp2Sep_deriv_gamma_k;
  } else if (kernel_type == "matern_3_2"){
    kernel_func = Matern_3_2_Sep;
    kernel_func_deriv_gamma_k = Matern_3_2_Sep_deriv_gamma_k;
  } else if (kernel_type == "matern_5_2"){
    kernel_func = Matern_5_2_Sep;
    kernel_func_deriv_gamma_k = Matern_5_2_Sep_deriv_gamma_k;
  } else {
    Rcpp::stop("Unknown kernel type.");
  }
  VectorXd grad;
  MatrixXd fisher; 
  if (nugget_est){
    int p = params.size()-1;
    VectorXd gamma = params.head(p);
    nu = params(p);
    grad = VectorXd::Zero(p+1);
    fisher = MatrixXd::Zero(p+1,p+1);
    if (zero_mean == "No"){
      MatrixXd H = trend;
      int q = H.cols();
      //neen special case for i = 0
      MatrixXd XSX = MatrixXd::Zero(q, q);
      
      // need to first compute this
      vector<vector<MatrixXd>> Bi_inv_B_ir_list(n, vector<MatrixXd>(p+1));
      vector<vector<MatrixXd>> Ai_inv_A_ir_list(n, vector<MatrixXd>(p+1));
      vector<MatrixXd> Ri_Bi_inv_list(n);
      vector<MatrixXd> Qi_Ai_inv_list(n);
      vector<MatrixXd> dXSXr(p+1, MatrixXd::Zero(q,q));
      vector<double> dlogdetr(p+1, 0.0);
      vector<VectorXi> idx_list(n);
      
      // deal with i = 0 separetely, only B_i
      
      for (int i = 0; i < n; ++i) {
        if (i == 0){
          MatrixXd v_i = x.row(i);
          MatrixXd R_i = H.row(i);
          double B_i = kernel_func(v_i, v_i, gamma, nu, alpha)(0,0);
          MatrixXd Ri_Bi_inv = (R_i * 1/B_i).transpose();
          Ri_Bi_inv_list[i] = Ri_Bi_inv;
          Qi_Ai_inv_list[i] = MatrixXd::Zero(q,1);
          XSX += Ri_Bi_inv * R_i;
          for (int r = 0; r < p; ++r){
            MatrixXd B_ir = kernel_func_deriv_gamma_k(v_i, v_i, gamma, nu, alpha, r);
            MatrixXd Bi_inv_B_ir = B_ir / B_i;
            dXSXr[r] -= Ri_Bi_inv * (Bi_inv_B_ir * R_i);
            dlogdetr[r] += Bi_inv_B_ir.trace();
            Bi_inv_B_ir_list[i][r] = Bi_inv_B_ir;
            Ai_inv_A_ir_list[i][r] = MatrixXd::Zero(1,1);
          }
          dXSXr[p] -= Ri_Bi_inv * Ri_Bi_inv.transpose();
          MatrixXd Bi_inv(1,1);
          Bi_inv(0,0) = 1/B_i;
          dlogdetr[p] += 1/B_i;
          // the last column saves Bi_inv;
          Bi_inv_B_ir_list[i][p] = Bi_inv;
          Ai_inv_A_ir_list[i][p] = MatrixXd::Zero(1,1);
          
          for (int r = 0; r < p + 1; ++r){
            for (int s = 0; s < p + 1; ++s){
              fisher(r,s) += 0.5 * (Bi_inv_B_ir_list[i][r] * Bi_inv_B_ir_list[i][s]).trace();
            }
          }
        } else{
          int num_neighbors = std::min(i, m);
          MatrixXd v_i(num_neighbors+1, p);
          v_i.row(0) = x.row(i);
          MatrixXd R_i(num_neighbors+1, q);
          R_i.row(0) = H.row(i);
          VectorXi idx = NNmatrix.block(i, 1, 1, num_neighbors).transpose().cast<int>();
          for (int j = 0; j < num_neighbors; ++j){
            v_i.row(j+1) = x.row(idx[j] - 1);
            R_i.row(j+1) = H.row(idx[j] - 1);
          }
          MatrixXd u_i = v_i.block(1, 0, num_neighbors, p);
          MatrixXd B_i = kernel_func(v_i, v_i, gamma, nu, alpha);
          // remove the first column and the first row;
          MatrixXd A_i = B_i.block(1, 1, num_neighbors, num_neighbors);
          MatrixXd Q_i = R_i.block(1, 0, num_neighbors, q);
          
          LLT<MatrixXd> llt_B_i(B_i); 
          LLT<MatrixXd> llt_A_i(A_i);
          MatrixXd Ri_Bi_inv = llt_B_i.solve(R_i).transpose();
          Ri_Bi_inv_list[i] = Ri_Bi_inv;
          MatrixXd Qi_Ai_inv = llt_A_i.solve(Q_i).transpose();
          Qi_Ai_inv_list[i] = Qi_Ai_inv;
          // done with XSX
          XSX += Ri_Bi_inv * R_i - Qi_Ai_inv* Q_i;
          
          // deal with gamma derive first
          for (int r = 0; r < p; ++r){
            MatrixXd B_ir = kernel_func_deriv_gamma_k(v_i, v_i, gamma, nu, alpha, r);
            MatrixXd A_ir = B_ir.block(1, 1, num_neighbors, num_neighbors);
            MatrixXd Bi_inv_B_ir = llt_B_i.solve(B_ir);
            MatrixXd Ai_inv_A_ir = llt_A_i.solve(A_ir);
            dXSXr[r] -= Ri_Bi_inv * (Bi_inv_B_ir * R_i) - Qi_Ai_inv * (Ai_inv_A_ir * Q_i);
            dlogdetr[r] += Bi_inv_B_ir.trace() - Ai_inv_A_ir.trace();
            Bi_inv_B_ir_list[i][r] = Bi_inv_B_ir;
            Ai_inv_A_ir_list[i][r] = Ai_inv_A_ir;
          }
          // now nu derive Bi_inv
          // done with dXSXr
          dXSXr[p] -= Ri_Bi_inv * Ri_Bi_inv.transpose() - Qi_Ai_inv * Qi_Ai_inv.transpose();
          MatrixXd Bi_inv = llt_B_i.solve(MatrixXd::Identity(num_neighbors+1, num_neighbors+1));
          MatrixXd Ai_inv = llt_A_i.solve(MatrixXd::Identity(num_neighbors, num_neighbors));
          // done with logdetr
          dlogdetr[p] += Bi_inv.trace() - Ai_inv.trace();
          // the last column saves Bi_inv;
          Bi_inv_B_ir_list[i][p] = Bi_inv;
          Ai_inv_A_ir_list[i][p] = Ai_inv;
          // construct the fisher information matrix
          for (int r = 0; r < p + 1; ++r){
            for (int s = 0; s < p + 1; ++s){
              fisher(r,s) += 0.5 * ((Bi_inv_B_ir_list[i][r] * Bi_inv_B_ir_list[i][s]).trace() -
                (Ai_inv_A_ir_list[i][r] * Ai_inv_A_ir_list[i][s]).trace());
            }
          }
        }
      }
      LLT<MatrixXd> llt_XSX(XSX);
      MatrixXd XSX_inv = llt_XSX.solve(MatrixXd::Identity(q, q));
      
      // finish the fisher information matrix
      fisher *= k;
      
      // deal with gamma and nu grad together
      for (int j = 0; j < k; ++j){
        double YSY_j = 0.0;
        VectorXd dYSYr_j = VectorXd::Zero(p+1);
        VectorXd XSY_j = VectorXd::Zero(q);
        vector<VectorXd> dXSYr_j(p+1, VectorXd::Zero(q));
        
        VectorXd y_j = y.col(j);
        for (int i = 1; i < n; ++i){
          if (i == 0){
            VectorXd y_v_i(1);
            y_v_i[0] = y_j[0];
            VectorXd Bi_inv_y_v_i = Bi_inv_B_ir_list[i][p] * y_v_i;
            YSY_j += y_v_i.dot(Bi_inv_y_v_i);
            XSY_j += Ri_Bi_inv_list[i] * y_v_i;
            for (int r = 0; r < p+1; ++r){
              dYSYr_j[r] -= y_v_i.dot(Bi_inv_B_ir_list[i][r] * (Bi_inv_y_v_i));
              dXSYr_j[r] -= Ri_Bi_inv_list[i] * (Bi_inv_B_ir_list[i][r].transpose() * y_v_i);
            }
          } else{
            int num_neighbors = std::min(i, m);
            VectorXi idx = NNmatrix.block(i, 1, 1, num_neighbors).transpose().cast<int>();
            // construct yv_i and yu_i
            VectorXd y_v_i(num_neighbors + 1);
            y_v_i[0] = y_j[i];
            for (int s = 0; s < num_neighbors; ++s){
              y_v_i[s+1] = y_j[idx[s] - 1];
            }
            VectorXd y_u_i = y_v_i.tail(num_neighbors);
            
            VectorXd Bi_inv_y_v_i = Bi_inv_B_ir_list[i][p] * y_v_i;
            VectorXd Ai_inv_y_u_i = Ai_inv_A_ir_list[i][p] * y_u_i;
            YSY_j += y_v_i.dot(Bi_inv_y_v_i) - y_u_i.dot(Ai_inv_y_u_i);
            XSY_j += Ri_Bi_inv_list[i] * y_v_i - Qi_Ai_inv_list[i] * y_u_i;
            
            for (int r = 0; r < p+1; ++r){
              dYSYr_j[r] -= y_v_i.dot(Bi_inv_B_ir_list[i][r] * (Bi_inv_y_v_i)) -
                y_u_i.dot(Ai_inv_A_ir_list[i][r] * (Ai_inv_y_u_i));
              dXSYr_j[r] -= Ri_Bi_inv_list[i] * (Bi_inv_B_ir_list[i][r].transpose() * y_v_i) - 
                Qi_Ai_inv_list[i] * (Ai_inv_A_ir_list[i][r].transpose() * y_u_i);
            }
          }
        }
        VectorXd XSX_inv_XSY_j = XSX_inv * XSY_j;
        // construct grad_j
        for (int r = 0; r < p + 1; ++r){
          double temp1 = dYSYr_j[r] - 2 * dXSYr_j[r].transpose() * XSX_inv_XSY_j + 
            XSY_j.transpose() * XSX_inv * dXSXr[r] * XSX_inv_XSY_j;
          double temp2 = YSY_j - XSY_j.transpose() * XSX_inv_XSY_j;
          double gradr_j = -n/2.0 * temp1 / temp2;
          grad[r] += gradr_j;
        }
      }
      // add logdetr to the full gradient
      for (int r = 0; r < p+1; ++r){
        grad[r] -= k/2.0 * dlogdetr[r];
      }
    }
  }
  auto end = high_resolution_clock::now();
  duration<double> diff = end - start;
  Rcpp::Rcout << "Fisher Scoring took " << diff.count() << " seconds.\n";
  return Rcpp::List::create(grad, fisher);
}

// [[Rcpp::export]]
MatrixXd Chol(const MatrixXd &R){
  
  LLT<MatrixXd> lltOfR(R);             // compute the cholesky decomposition of R called lltofR
  MatrixXd L = lltOfR.matrixL();   //retrieve factor L  in the decomposition
  return L;
}

// [[Rcpp::export]]
Eigen::MatrixXd euclidean_distance(const MatrixXd & input1,const MatrixXd & input2){
  //input are n by p, where p is larger than n
  
  int num_obs1 = input1.rows();
  int num_obs2 = input2.rows();
  
  Eigen::MatrixXd R0=R0.Ones(num_obs1,num_obs2);
  
  for (int i = 0; i < num_obs1; i++){
    
    for (int j = 0; j < num_obs2; j++){
      R0(i,j)=sqrt((input1.row(i)-input2.row(j)).array().pow(2.0).sum());
    }
  }
  return R0;
}

// [[Rcpp::export]]
List generate_R0(const MatrixXd & input1, const MatrixXd & input2){
  int n = input1.rows();
  int p = input2.cols();
  List result(p);
  for (int i = 0; i < p; i++){
    MatrixXd R0 = MatrixXd::Zero(n, n);
    for (int j = 0; j < n; j++){
      for (int z = 0; z < n; z++){
        R0(j,z) = abs(input1(j,i) - input2(z,i));
      }
    }
    result[i] = R0;
  }
  return result;
}

// [[Rcpp::export]]
Eigen::MatrixXd pow_exp_funct (const MatrixXd &d, double beta_i,double alpha_i){
  
  return (-(beta_i*d).array().pow(alpha_i)).exp().matrix();
}

// [[Rcpp::export]]
Eigen::MatrixXd separable_multi_kernel (List R0, const Eigen::VectorXd  & beta,const Eigen::VectorXi  & kernel_type,const Eigen::VectorXd  & alpha ){
  Eigen::MatrixXd R0element = R0[0];
  int Rnrow = R0element.rows();
  int Rncol = R0element.cols();
  
  Eigen::MatrixXd R = R.Ones(Rnrow,Rncol);
  //String kernel_type_i_ker;
  for (int i_ker = 0; i_ker < beta.size(); i_ker++){
    if(kernel_type[i_ker]==1){
      R = (pow_exp_funct(R0[i_ker],beta[i_ker],alpha[i_ker])).cwiseProduct(R);
    }
  }
  return R;
}

// [[Rcpp::export]]
double log_marginal_lik_ppgasp(const Eigen::VectorXd &  param,double nugget, 
                               const bool nugget_est, const List R0, 
                               const Eigen::MatrixXd & X,
                               const String zero_mean,
                               const Eigen::MatrixXd & output, 
                               const Eigen::VectorXi  &kernel_type,
                               const Eigen::VectorXd & alpha ){
  Eigen::VectorXd beta;
  double nu=nugget;
  int k=output.cols();
  int param_size=param.size();
  if(!nugget_est){
    beta= param.array().exp().matrix();
    // nu=0;
  }else{
    beta=param.head(param_size-1).array().exp().matrix(); 
    nu=exp(param[param_size-1]); //nugget
  }
  
  int num_obs=output.rows();
  MatrixXd R= separable_multi_kernel(R0,beta, kernel_type,alpha);
  R=R+nu*MatrixXd::Identity(num_obs,num_obs);  //not sure 
  
  LLT<MatrixXd> lltOfR(R);             // compute the cholesky decomposition of R called lltofR
  MatrixXd L = lltOfR.matrixL();   //retrieve factor L  in the decomposition
  if(zero_mean=="Yes"){
    
    MatrixXd yt_R_inv= (L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(output))).transpose(); 
    
    
    
    double log_S_2=0;
    
    for(int loc_i=0;loc_i<k;loc_i++){
      log_S_2=log_S_2+log((yt_R_inv.row(loc_i)*output.col(loc_i))(0,0));
    }
    
    //double log_S_2=log(S_2);
    
    return -k*(L.diagonal().array().log().matrix().sum())-(num_obs)/2.0*log_S_2;
    
  }else{
    
    int q=X.cols();
    
    MatrixXd R_inv_X=L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(X)); //one forward and one backward to compute R.inv%*%X
    MatrixXd Xt_R_inv_X=X.transpose()*R_inv_X; //Xt%*%R.inv%*%X
    
    LLT<MatrixXd> lltOfXRinvX(Xt_R_inv_X); // cholesky decomposition of Xt_R_inv_X called lltOfXRinvX
    MatrixXd LX = lltOfXRinvX.matrixL();  //  retrieve factor LX  in the decomposition 
    MatrixXd R_inv_X_Xt_R_inv_X_inv_Xt_R_inv= R_inv_X*(LX.transpose().triangularView<Upper>().solve(LX.triangularView<Lower>().solve(R_inv_X.transpose())));          //compute  R_inv_X_Xt_R_inv_X_inv_Xt_R_inv through one forward and one backward solve
    MatrixXd yt_R_inv= (L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(output))).transpose(); 
    
    //double log_S_2=((yt_R_inv.array()*output.array()).rowwise().sum().log()).sum();
    
    
    double log_S_2=0;
    
    for(int loc_i=0;loc_i<k;loc_i++){
      log_S_2=log_S_2+log((yt_R_inv.row(loc_i)*output.col(loc_i))(0,0)-(output.col(loc_i).transpose()*R_inv_X_Xt_R_inv_X_inv_Xt_R_inv*output.col(loc_i))(0,0));
    }
    
    MatrixXd R_inv = lltOfR.solve(MatrixXd::Identity(R.rows(), R.cols()));
    // double log_S_2=log(S_2);
    MatrixXd Q = R_inv -  R_inv_X_Xt_R_inv_X_inv_Xt_R_inv;
    //MatrixXd S_2= (yt_R_inv*output-output.transpose()*R_inv_X_Xt_R_inv_X_inv_Xt_R_inv*output);
    //double log_S_2=log(S_2(0,0));
    return (-k*(L.diagonal().array().log().matrix().sum())-k*(LX.diagonal().array().log().matrix().sum())-(num_obs-q)/2.0*log_S_2);
  }
}

// [[Rcpp::export]]
Eigen::MatrixXd pow_exp_deriv(const MatrixXd & R0_i, const Eigen::MatrixXd & R, const double beta_i, const double alpha_i){
  return  -(R.array()*(R0_i.array().pow(alpha_i))).matrix()*alpha_i*pow(beta_i,alpha_i-1);
}

// [[Rcpp::export]]
List log_marginal_lik_deriv_ppgasp(const VectorXd & param,
                                              double nugget,  
                                              bool nugget_est, 
                                              const List R0, 
                                              const MatrixXd& X,
                                              const String zero_mean,
                                              const MatrixXd & output, 
                                              const VectorXi & kernel_type,
                                              const VectorXd & alpha){
  
  VectorXd beta;
  double nu=nugget;
  int k=output.cols();
  int param_size=param.size();
  if(nugget_est==false){//not sure about the logical stuff
    beta= param.array().exp().matrix();
  }else{
    beta=param.head(param_size-1).array().exp().matrix(); 
    nu=exp(param[param_size-1]); //nugget
  }
  int p=beta.size();
  int num_obs=output.rows();
  MatrixXd R= separable_multi_kernel(R0,beta,kernel_type,alpha);
  MatrixXd R_ori=  R;  // this is the one without the nugget
  
  R=R+nu*MatrixXd::Identity(num_obs,num_obs);  //not sure 
  
  LLT<MatrixXd> lltOfR(R);
  MatrixXd L = lltOfR.matrixL();
  VectorXd ans=VectorXd::Ones(param_size);
  
  //String kernel_type_ti;
  
  MatrixXd Vb_ti;
  if(zero_mean=="Yes"){
    MatrixXd yt_R_inv= (L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(output))).transpose(); 
    //MatrixXd S_2= (yt_R_inv*output);
    
    //double log_S_2=log(S_2(0,0));
    VectorXd S_2_vec=VectorXd::Zero(k);
    
    for(int loc_i=0;loc_i<k;loc_i++){
      S_2_vec[loc_i]=(yt_R_inv.row(loc_i)*output.col(loc_i))(0,0);
      
    }
    MatrixXd dev_R_i;
    //allow different choices of kernels
    for(int ti=0;ti<p;ti++){
      if(kernel_type[ti]==1){
        dev_R_i=pow_exp_deriv( R0[ti],R_ori,beta[ti],alpha[ti]);   //now here I have R_ori instead of R
      }
      Vb_ti=L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(dev_R_i));
      
      
      double ratio=0;
      
      for(int loc_i=0;loc_i<k;loc_i++){
        ratio=ratio+((output.col(loc_i).transpose()*Vb_ti*(yt_R_inv.transpose()).col(loc_i) )(0,0))/S_2_vec[loc_i];
      }
      ans[ti]=-0.5*k*Vb_ti.diagonal().sum()+num_obs/2.0*ratio;
      //ans[ti]=-0.5*Vb_ti.diagonal().sum()+(num_obs/2.0*output.transpose()*Vb_ti*yt_R_inv.transpose()/ S_2(0,0))(0,0) ;  
    }
    //the last one if the nugget exists
    if(nugget_est){
      dev_R_i=MatrixXd::Identity(num_obs,num_obs);
      Vb_ti=L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(dev_R_i));
      
      double ratio=0;
      for(int loc_i=0;loc_i<k;loc_i++){
        ratio=ratio+((output.col(loc_i).transpose()*Vb_ti*(yt_R_inv.transpose()).col(loc_i))(0,0))/S_2_vec[loc_i];
      }
      ans[p]=-0.5*k*Vb_ti.diagonal().sum()+num_obs/2.0*ratio;
      //ans[p]=-0.5*Vb_ti.diagonal().sum()+(num_obs/2.0*output.transpose()*Vb_ti*yt_R_inv.transpose()/ S_2(0,0))(0,0); 
    }
  }else{
    int q=X.cols();
    MatrixXd R_inv_X=L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(X));
    MatrixXd Xt_R_inv_X=X.transpose()*R_inv_X;
    
    LLT<MatrixXd> lltOfXRinvX(Xt_R_inv_X);
    MatrixXd LX = lltOfXRinvX.matrixL();
    MatrixXd R_inv_X_Xt_R_inv_X_inv_Xt_R_inv= R_inv_X*(LX.transpose().triangularView<Upper>().solve(LX.triangularView<Lower>().solve(R_inv_X.transpose())));
    MatrixXd yt_R_inv= (L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(output))).transpose();
    MatrixXd dev_R_i;
    MatrixXd Wb_ti;
    //allow different choices of kernels
    
    
    VectorXd S_2_vec=VectorXd::Zero(k);
    
    for(int loc_i=0;loc_i<k;loc_i++){
      S_2_vec[loc_i]=(yt_R_inv.row(loc_i)*output.col(loc_i))(0,0)-(output.col(loc_i).transpose()*R_inv_X_Xt_R_inv_X_inv_Xt_R_inv*output.col(loc_i))(0,0);
    }
    
    
    // double log_S_2=0;
    
    //for(int loc_i=0;loc_i<k;loc_i++){
    //  log_S_2=log_S_2+log((yt_R_inv.row(loc_i)*output.col(loc_i))(0,0)-(output.col(loc_i).transpose()*R_inv_X_Xt_R_inv_X_inv_Xt_R_inv*output.col(loc_i))(0,0));
    //}
    
    
    for(int ti=0;ti<p;ti++){
      //kernel_type_ti=kernel_type[ti];
     if(kernel_type[ti]==1){
        dev_R_i=pow_exp_deriv( R0[ti],R_ori,beta[ti],alpha[ti]);   //now here I have R_ori instead of R
      }
      Wb_ti=(L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(dev_R_i))).transpose()-dev_R_i*R_inv_X_Xt_R_inv_X_inv_Xt_R_inv;
      
      double ratio=0;
      
      for(int loc_i=0;loc_i<k;loc_i++){
        ratio=ratio+((output.col(loc_i).transpose()*Wb_ti.transpose()*(yt_R_inv.row(loc_i).transpose()-R_inv_X_Xt_R_inv_X_inv_Xt_R_inv*output.col(loc_i)))(0,0))/S_2_vec[loc_i];
      }
      
      
      
      //MatrixXd S_2= (yt_R_inv*output-output.transpose()*R_inv_X_Xt_R_inv_X_inv_Xt_R_inv*output);
      
      //MatrixXd Q_output= yt_R_inv.transpose()-R_inv_X_Xt_R_inv_X_inv_Xt_R_inv*output;
      
      ans[ti]=-0.5*k*Wb_ti.diagonal().sum()+(num_obs-q)/2.0*ratio; 
      
      
      //ans[ti]=-0.5*Wb_ti.diagonal().sum()+(num_obs-q)/2.0*(output.transpose()*Wb_ti.transpose()*Q_output/S_2(0,0))(0,0); 
    }
    
    
    
    if(nugget_est){
      dev_R_i=MatrixXd::Identity(num_obs,num_obs);
      Wb_ti=(L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(dev_R_i))).transpose()-dev_R_i*R_inv_X_Xt_R_inv_X_inv_Xt_R_inv;
      
      //double S_2_dev=0;
      double ratio=0;
      
      for(int loc_i=0;loc_i<k;loc_i++){
        ratio=ratio+((output.col(loc_i).transpose()*Wb_ti.transpose()*(yt_R_inv.row(loc_i).transpose()-R_inv_X_Xt_R_inv_X_inv_Xt_R_inv*output.col(loc_i)))(0,0))/S_2_vec[loc_i];
      }
      ans[p]=-0.5*k*Wb_ti.diagonal().sum()  +(num_obs-q)/2.0*ratio; 
      
      //ans[p]=-0.5*Wb_ti.diagonal().sum()+(num_obs-q)/2.0*(output.transpose()*Wb_ti.transpose()*Q_output/S_2(0,0))(0,0); 
    }
    
    
  }
  return Rcpp::List::create(ans, Vb_ti);
}

