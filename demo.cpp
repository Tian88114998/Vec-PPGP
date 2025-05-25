
// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*- 
 
// we only include RcppEigen.h which pulls Rcpp.h in for us 
#include <Rcpp.h> 
#include <RcppEigen.h> 
#include <cmath> 
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
      double r = 0.0;
      for (int k = 0; k < p; k++){
        // change to squared value instead of absolute value
        r += pow((std::abs(x1(i, k) - x2(j, k)) / gamma(k)), alpha);
      }
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
      double extra = alpha * 
        pow(gamma(k), - alpha - 1) * pow(abs(x1(i,k)-x2(j,k)), alpha);
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

//[[Rcpp::export]]
double neg_vecchia_marginal_log_likelihood(const VectorXd params, double nu, const bool nugget_est, 
                                           const MatrixXd &x, const MatrixXd &NNmatrix, const MatrixXd &y,
                                           const MatrixXd trend, const String kernel_type, 
                                           const double alpha, const String zero_mean = "Yes"){
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
    MatrixXd H = trend;
    MatrixXd N = H.transpose() * UUt * H;
    LLT<MatrixXd> llt(N);
    MatrixXd N_inv_H_t = llt.solve(H.transpose());
    MatrixXd Q = UUt * (MatrixXd::Identity(n,n) - H * N_inv_H_t * UUt);
    MatrixXd L = llt.matrixL();
    double N_log_det = L.diagonal().array().log().matrix().sum();
    for (int i = 0; i < k; ++i) {
      res += log(y.col(i).transpose() * Q * y.col(i));
    }
    // delete by 2.0 but not 2
    res *= -(n-q)/2.0;
    res += k * sum_log_diag_U;
    res -= k * N_log_det;
  }
  return -res;
}

// TODO: check if this function is correct. 
// [[Rcpp::export]]
VectorXd neg_vecchia_marginal_log_likelihood_deriv(const VectorXd params, double nu, const bool nugget_est,
                                                   const MatrixXd &x, const MatrixXd &NNmatrix,
                                                   const MatrixXd &y, const MatrixXd &trend,
                                                   const String kernel_type, const double alpha, 
                                                   const String zero_mean = "No"){
  // if nugget_est = TRUE, params should contain gamma and nu, and use the nu in params
  // if nuggest_est = FALSE, params should contain only gamma, and use the nu as given.
  // y is n * k 
  
  //TODO: add mean
  int k = y.cols();
  int n = y.rows();
  int m = NNmatrix.cols() - 1;
  MatrixXd H = trend;
  MatrixXd U = MatrixXd::Zero(n,n);
  std::function<MatrixXd(const MatrixXd&, const MatrixXd&, const VectorXd&, double, double)> kernel_func;
  std::function<MatrixXd(const MatrixXd&, const MatrixXd&, const VectorXd&, double, double, int)> kernel_func_deriv_gamma_k;
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
    int p = params.size()-1;
    VectorXd gamma = params.head(p);
    nu = params(p);
    U(0,0) = 1/sqrt(kernel_func(x.row(0),x.row(0), gamma, nu, alpha)(0,0));
  
    VectorXd grad = VectorXd::Zero(p+1);
    VectorXd sum_log_deriv = VectorXd::Zero(p+1);
    
    // a list of matrix to store dU/dgamma_k and dU/dnu 
    std::vector<MatrixXd> matrix_list(p+1, MatrixXd::Zero(n, n));
    // first set the (1,1) element of dU/dnu
    matrix_list[p](0,0) = -0.5 * pow(kernel_func(x.row(0), x.row(0), gamma, nu, alpha)(0,0), -3/2);
    // the (1,1) element of du/dgamma is 0
    
    sum_log_deriv(p) -= 0.5 * 1/kernel_func(x.row(0), x.row(0), gamma, nu, alpha)(0,0);
    // deal with i = 0 case separately since there is no neighbor
    for (int i = 1; i < n; ++i){
      MatrixXd selected_neighbors;
      VectorXd idx;
      if (i < m) {
        idx = NNmatrix.block(i, 1, 1, i).transpose();
        selected_neighbors.resize(i, x.cols());
        // select the neighbors of i according to the idx set for later use
        for (int j = 0; j < i; ++j){
          selected_neighbors.row(j) = x.row(idx[j] - 1);
        }
        // also calculate the Umatrix
        List temp = Uentries(x, idx, i, gamma, nu, kernel_type, alpha);
        VectorXd icolumn = temp[0];
        VectorXd ucol = Ucolumn(icolumn, idx, i, n);
        U.col(i) = ucol;
      } else{
        idx = NNmatrix.row(i).segment(1, m);
        selected_neighbors.resize(m, x.cols());
        // select the neighbors of i according to the idx set
        for (int j = 0; j < m; ++j){
          selected_neighbors.row(j) = x.row(idx[j] - 1);
        }
        List temp = Uentries(x, idx, i, gamma, nu, kernel_type, alpha);
        VectorXd icolumn = temp[0];
        VectorXd ucol = Ucolumn(icolumn, idx, i, n);
        U.col(i) = ucol;
      }
      
      // first calculate the gradient of gamma
      MatrixXd i_element = x.row(i);
      MatrixXd Sigma_cc = kernel_func(selected_neighbors, selected_neighbors, gamma, nu, alpha);
      MatrixXd Sigma_ci = kernel_func(selected_neighbors, i_element, gamma, nu, alpha);
      MatrixXd B_i = R_inv_y(Sigma_cc, Sigma_ci).transpose();
      MatrixXd D_i = kernel_func(i_element, i_element, gamma, nu,
                             alpha) - B_i * Sigma_ci;
      // this is 1/sigma
      double D_i_half_inv = 1/sqrt(D_i(0,0));
      
      
      for (int j = 0; j < p; ++j){
        MatrixXd Sigma_ci_deriv_gamma_j = kernel_func_deriv_gamma_k(selected_neighbors, i_element, gamma, nu, alpha, j);
        MatrixXd Sigma_cc_deriv_gamma_j = kernel_func_deriv_gamma_k(selected_neighbors, selected_neighbors, gamma, nu, alpha, j);
        MatrixXd temp1 = Sigma_ci_deriv_gamma_j.transpose() * B_i.transpose() - B_i * Sigma_cc_deriv_gamma_j * B_i.transpose() + 
          B_i * Sigma_ci_deriv_gamma_j;
        double A_j = temp1(0,0);
        // this finishes the first part
        sum_log_deriv(j) += 0.5 * pow(D_i_half_inv, 2) * A_j;
        // the second part we need to dU/dgamma
        double diag_j = 0.5 * pow(D_i_half_inv, 3) * A_j;
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
      
      double W = 1.0 + (B_i * B_i.transpose())(0,0);
      sum_log_deriv(p) -= 0.5 * pow(D_i_half_inv,2) * W;
      double diag_nu = -0.5 * pow(D_i_half_inv, 3) * W;
      MatrixXd off_diag_nu = -diag_nu * B_i + D_i_half_inv * R_inv_y(Sigma_cc, B_i.transpose()).transpose();
      // get the i_column of dU/dnu
      VectorXd i_column(off_diag_nu.cols() + 1);
      i_column.head(off_diag_nu.cols()) = off_diag_nu;
      i_column(off_diag_nu.cols()) = diag_nu;
      VectorXd ucol = Ucolumn(i_column, idx, i, n);
      matrix_list[p].col(i) = ucol;
    }
    
    MatrixXd Ut = U.transpose();
    MatrixXd UUt = U*Ut;
    if (zero_mean == "Yes"){
      // here rewrite the for loop to make it faster.
      for (int j = 0; j < p+1; ++j){
        grad(j) -= k * sum_log_deriv(j);
        for (int z = 0; z < k; ++z) {
          MatrixXd temp3 =  y.col(z).transpose() * UUt * y.col(z);
          MatrixXd temp4 = y.col(z).transpose() * matrix_list[j] * Ut * y.col(z);
          grad(j) += n * 1/temp3(0,0) * temp4(0,0);
          if (j == 2){
            Rcpp::Rcout << 1/temp3(0,0) * temp4(0,0) << std::endl;
          }
        }
      }
    } else{
      int q = y.cols();
      MatrixXd Ht = H.transpose();
      MatrixXd N = Ht * UUt * H;
      LLT<MatrixXd> llt(N);
      MatrixXd N_inv = llt.solve(MatrixXd::Identity(q, q));
      MatrixXd temp3 = H * N_inv * Ht;
      MatrixXd temp4 = MatrixXd::Identity(n,n) - temp3 * UUt;
      MatrixXd Q = UUt * temp4;
      MatrixXd temp5 = H * N * Ht;
      for (int j = 0; j < p+1; ++j){
        MatrixXd M = matrix_list[j] * Ut;
        MatrixXd dQ = M * temp4 + UUt * (temp3 * M - temp5 * M * temp5 * UUt);
        for (int z = 0; z < k; ++z) {
          MatrixXd temp6 = y.col(z).transpose() * Q * y.col(z);
          MatrixXd temp7 = y.col(z).transpose() * dQ * y.col(z);
          grad(j) += (n-q) * 1/temp6(0,0) * temp7(0,0);
        }
      }
    }
    return grad;
  } else{
    int p = params.size();
    VectorXd gamma = params;
    U(0,0) = 1/sqrt(kernel_func(x.row(0),x.row(0), gamma, nu, alpha)(0,0));
    
    VectorXd grad = VectorXd::Zero(p);
    VectorXd sum_log_deriv = VectorXd::Zero(p);
    std::vector<MatrixXd> matrix_list(p, MatrixXd::Zero(n, n));
    
    // deal with i = 0 case separately since there is no neighbor
    for (int i = 1; i < n; ++i){
      MatrixXd selected_neighbors;
      VectorXd idx;
      if (i < m) {
        idx = NNmatrix.block(i, 1, 1, i).transpose();
        selected_neighbors.resize(i, x.cols());
        // select the neighbors of i according to the idx set for later use
        for (int j = 0; j < i; ++j){
          selected_neighbors.row(j) = x.row(idx[j] - 1);
        }
        // also calculate the Umatrix
        List temp = Uentries(x, idx, i, gamma, nu, kernel_type, alpha);
        VectorXd icolumn = temp[0];
        VectorXd ucol = Ucolumn(icolumn, idx, i, n);
        U.col(i) = ucol;
      } else{
        idx = NNmatrix.row(i).segment(1, m);
        selected_neighbors.resize(m, x.cols());
        // select the neighbors of i according to the idx set
        for (int j = 0; j < m; ++j){
          selected_neighbors.row(j) = x.row(idx[j] - 1);
        }
        List temp = Uentries(x, idx, i, gamma, nu, kernel_type, alpha);
        VectorXd icolumn = temp[0];
        VectorXd ucol = Ucolumn(icolumn, idx, i, n);
        U.col(i) = ucol;
      }
      
      // first calculate the gradient of gamma
      MatrixXd i_element = x.row(i);
      MatrixXd Sigma_cc = kernel_func(selected_neighbors, selected_neighbors, gamma, nu, alpha);
      MatrixXd Sigma_ci = kernel_func(selected_neighbors, i_element, gamma, nu, alpha);
      MatrixXd B_i = R_inv_y(Sigma_cc, Sigma_ci).transpose();
      MatrixXd D_i = kernel_func(i_element, i_element, gamma, nu,
                             alpha) - B_i * Sigma_ci;
      // this is 1/sigma
      double D_i_half_inv = 1/sqrt(D_i(0,0));
      
      
      for (int j = 0; j < p; ++j){
        MatrixXd Sigma_ci_deriv_gamma_j = kernel_func_deriv_gamma_k(selected_neighbors, i_element, gamma, nu, alpha, j);
        MatrixXd Sigma_cc_deriv_gamma_j = kernel_func_deriv_gamma_k(selected_neighbors, selected_neighbors, gamma, nu, alpha, j);
        MatrixXd temp1 = Sigma_ci_deriv_gamma_j.transpose() * B_i.transpose() - B_i * Sigma_cc_deriv_gamma_j * B_i.transpose() + 
          B_i * Sigma_ci_deriv_gamma_j;
        double A_j = temp1(0,0);
        // this finishes the first part
        sum_log_deriv(j) += 0.5 * pow(D_i_half_inv, 2) * A_j;
        // the second part we need to dU/dgamma
        double diag_j = 0.5 * pow(D_i_half_inv, 3) * A_j;
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
    // finish the gradient of gamma and nu
    MatrixXd Ut = U.transpose();
    MatrixXd UUt = U*Ut;
    for (int j = 0; j < p; ++j){
      grad(j) -= k * sum_log_deriv(j);
      for (int z = 0; z < k; ++z) {
        MatrixXd temp3 =  y.col(z).transpose() * UUt * y.col(z);
        MatrixXd temp4 = y.col(z).transpose() * matrix_list[j] * Ut * y.col(z);
        grad(j) += n * 1/temp3(0,0) * temp4(0,0);
      }
    }
    return grad;
  }
}

// [[Rcpp::export]]
// make prediction on new points 
List predict(const MatrixXd &x, const MatrixXd &xp, const MatrixXd &NNmatrix,
               const MatrixXd &y, const VectorXd gamma, const double nu, 
               const String kernel_type, const double alpha, const double q)
{
  int k = y.cols();
  int n = y.rows();
  int np = xp.rows();
  MatrixXd allx(x.rows() + xp.rows(), x.cols());
  allx.topRows(x.rows()) = x;
  allx.bottomRows(xp.rows()) = xp;
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
  for (int j = 0; j < k; j++){
    MatrixXd y_j = y.col(j);
    double sigma2_j = (y_j.transpose() * UoUot * y_j)(0,0)/n;
    MatrixXd temp_mean = - Up.transpose().triangularView<Eigen::Lower>().solve(Uopt * y_j);
    pred_mean.col(j) = temp_mean;
    MatrixXd temp_sd = q * ((sigma2_j * Sigma_ii.array())).sqrt().matrix();
    MatrixXd temp_bound = temp_mean + temp_sd;
    upper95.col(j) = temp_bound;
    temp_bound = temp_mean - temp_sd;
    lower95.col(j) = temp_bound;
  }
  return Rcpp::List::create(pred_mean, lower95, upper95);
}

// [[Rcpp::export]]
double Test_for_loop(const int n){
  double res=0;
  if(n>0){
    for(int i=0; i<n;i++){
      res=res+0.1;
    }
  }
  return res;
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
Eigen::VectorXd log_marginal_lik_deriv_ppgasp(const VectorXd & param,
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
  
  if(zero_mean=="Yes"){
    MatrixXd yt_R_inv= (L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(output))).transpose(); 
    //MatrixXd S_2= (yt_R_inv*output);
    
    //double log_S_2=log(S_2(0,0));
    VectorXd S_2_vec=VectorXd::Zero(k);
    
    for(int loc_i=0;loc_i<k;loc_i++){
      S_2_vec[loc_i]=(yt_R_inv.row(loc_i)*output.col(loc_i))(0,0);
      
    }
    MatrixXd dev_R_i;
    MatrixXd Vb_ti;
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
        Rcpp::Rcout << ((output.col(loc_i).transpose()*Vb_ti*(yt_R_inv.transpose()).col(loc_i))(0,0))/S_2_vec[loc_i] << std::endl;
      }
      ans[p]=-0.5*k*Vb_ti.diagonal().sum()+num_obs/2.0*ratio;
      //ans[p]=-0.5*Vb_ti.diagonal().sum()+(num_obs/2.0*output.transpose()*Vb_ti*yt_R_inv.transpose()/ S_2(0,0))(0,0); 
    }
  }
  return ans;
}

