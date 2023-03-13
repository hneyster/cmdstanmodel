//  Multi-species Abundance Model (MSAM) to model the abundance of birds in Vancovuer in 2020
// using data replicated from melles (2000)
// Written by Harold Eyster 
// revised, added beta dist on p. 
// added 1997 bird observations 
// added scale selection 
// adding phi that relates present to past abundance 
functions {
  row_vector scale_select(real zeta, matrix E) {
    int index = cols(E);
    
    // If zeta is already above the maximum index then
    // we can skip the interpolation
    if (index < zeta) {
      return E[index,:];
    }
    
    while (index > zeta) {
      index -= 1;
    }
    
    real prop_scales = zeta - index;
    
    // Assumes same value of zeta for every row
    return E[index,:] * (1 - prop_scales) + E[index + 1,:] * prop_scales;
  }
}
data {
  int<lower=0> R;       // Number of sites in season 1
  int<lower=0> J;       // Number of temporal replicates in season 2
  int<lower=0> S;       // Number of species 
  int<lower=0> H;       // number of habitat types 
  int<lower=0> C;       // number of scales 
  array [R,S] int <lower=0> y1; // Counts in season 1
  array [R,J,S] int<lower=0> y2; // counts in season 2
  int<lower=0> K;       // Upper bound of population size
  array [R] matrix[C,H] E1;      // environmental covariates in season 1
  array [R] matrix [C,H] E2;       // environmental covariates in season 2 
}

transformed data {
  array [R,S] int<lower=0> max_y2; // the the max number of each sp observed at each spatial replication.
  array [R,S] int<lower=0> max_y1;
  max_y1 = y1; // no temporal replication, so it is the max 
  for (i in 1:R) {
    for (j in 1:S)
      max_y2[i,j] = max(y2[i,,j]); 
  }
  
//   array [R] matrix [C,H] E1_arr; 
//   array [R] matrix [C,H] E2_arr;  
//   for (j in 1:R){
//         E1_arr[j] = to_matrix(E1[j,:,:]);
//         E2_arr[j] = to_matrix(E2[j,:,:]);
//     }
}


parameters {
    vector <lower=1, upper = C>[S] zeta;
    real <lower=0, upper=1> psi;       // 
    real <lower=.1> theta; // 
    vector[H] mu;
    vector<lower=0>[H] tau;
    matrix [H,S] beta_tilde; // centered habitat weights 
    vector <lower=0,upper=1> [S] p;  // 
    real<lower=1, upper=C> kappa; 
    real<lower=0> omega;
    vector [S] phi; 
    real rho;
    real <lower=0> sigma; 
}
transformed parameters{
  matrix[H,S] beta; // habitat sensitivity by species 
  for (i in 1:H){
    for (j in 1:S){
    beta[i,j] = mu[i]+tau[i]*beta_tilde[i,j];
  //implies habitat ~ normal(mu_habitat, sig_habitat
    }
  }
  // implies p ~ logistic(mu_p, sig_p)
   
}

model {
  // Priors
  // priors on beta:
  mu[1] ~ normal(0,1); //water 
  mu[2] ~ normal(0,1); //buildings
  mu[3] ~ normal(0,1); //paved
  mu[4] ~ normal(0,1); //barren
  mu[5] ~ normal(0,1); // grass-herb
  mu[6] ~ normal(0,1); // coniferous 
  mu[7] ~ normal(0,1); // deciduous
  
  tau ~ normal(0,2);
  //priors on p:
  psi ~ beta(1,1); // following Gelmen et al 2013, ch. 5 
  theta ~ pareto(.1,1.5); // following Gelmen et al 2013, ch. 5 
  
  kappa ~ cauchy(1,C);
  omega ~ cauchy(0,10);
  
  rho ~ normal(0,10);
  sigma ~ normal(0,10);
  
  zeta ~ cauchy(kappa, omega); //uniform(1,C);
  phi ~ normal(rho, sigma);
  
  
  p ~ beta(theta*psi, theta*(1-psi));

  to_vector(beta_tilde) ~ std_normal(); //implies habitat ~ normal(mu, tau)

  // Likelihood
  for (i in 1:R) {
    for (s in 1:S){ 
      // season 2: 
      vector[K - max_y2[i,s] + 1] lp2; //it's the product, lambda*p that's calculated
        
      for (j in 1:(K - max_y2[i,s] + 1)){
           lp2[j] = poisson_log_lpmf(max_y2[i,s] + j - 1 | scale_select(zeta[s],E2[i])*beta[,s] + phi[s]*scale_select(zeta[s],E1[i])*beta[,s]) 
               + binomial_lpmf(y2[i,,s] | max_y2[i,s] + j - 1, p[s]); // implicitly sums across T (y[i] is vectorized)
      }
      target += log_sum_exp(lp2);
      // season 1
      vector[K - max_y1[i,s] + 1] lp1;

            for (j in 1:(K - max_y1[i,s] + 1)){
           lp1[j] = poisson_log_lpmf(max_y1[i,s] + j - 1 |scale_select(zeta[s],E1[i])*beta[,s]) 
               + binomial_lpmf(y1[i,s] | max_y1[i,s] + j - 1, p[s]); // implicitly sums across T (y[i] is vectorized)
      }
      target += log_sum_exp(lp1);
    }
  }
}

generated quantities{
  array [R,S] int N1;
  array [R,S] int N2;
 // int PPC[R,S];

  for (i in 1:R){
    for (s in 1:S){
      N1[i,s] = poisson_log_rng(scale_select(zeta[s],E1[i])*beta[,s]);
      N2[i,s] = poisson_log_rng(scale_select(zeta[s],E2[i])*beta[,s] + phi[s]*scale_select(zeta[s],E2[i])*beta[,s]);
    }
  }
}
