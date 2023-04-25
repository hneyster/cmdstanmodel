//  Multi-species Abundance Model (MSAM) to model the abundance of birds in Vancovuer in 2020
// using data replicated from melles (2000)
// Written by Harold Eyster 
// revised, added beta dist on p. 
// added 1997 bird observations 
// added phi, but as a multiplyer 
// adding optimum scales
// adding intercept
// adding finer priors 
data {
  int<lower=0> R;       // Number of sites in season 1
  int<lower=0> J;       // Number of temporal replicates in season 2
  int<lower=0> S;       // Number of species 
  int<lower=0> C;       // number of land cover scales 
  int<lower=0> H;       // number of habitat types  
  array [R,S] int<lower=0> y1; // Counts in season 1
  array [R,J,S] int<lower=0> y2; // counts in season 2
  int<lower=0> K;       // Upper bound of population size
  array [R] matrix [C,H]  E1;      // environmental covariates in season 1
  array [R] matrix [C,H] E2;       // environmental covariates in season 2 
  array [S, H] int <lower=1> opt;
}

transformed data {
  array [R,S] int<lower=0> max_y2; // the the max number of each sp observed at each spatial replication.
  array [R,S] int<lower=0> max_y1;
  max_y1 = y1; // no temporal replication, so it is the max 
  for (i in 1:R) {
    for (j in 1:S)
      max_y2[i,j] = max(y2[i,,j]); 
  }
  array [R] matrix[S,H+1]  E1_opt;
  array [R] matrix[S,H+1]  E2_opt;
  for (i in 1:S){
   for (j in 1: H){
     E1_opt[:,i,j] = to_array_1d(E1[,opt[i,j],j]);
     E2_opt[:,i,j] = to_array_1d(E2[,opt[i,j],j]);
     E1_opt[:,i,H+1] = rep_array(1,R);
     E2_opt[:,i,H+1] = rep_array(1,R);

   }
  }
}
parameters {
    real <lower=0, upper=1> psi;  // detection prob center parameter 
    real <lower=0> theta; // detection prob scale parameter 
    vector[H+1] mu; // land cover effect center paramters 
    vector<lower=0>[H+1] tau; // land cover scale parameters 
    matrix [H+1,S] beta_tilde; // centered habitat weights 
    vector <lower=0,upper=1> [S] p;  // centered detection variation by species 
    vector [S] phi_tilde;
    real rho; // change between season 1 and 2, center parameter
    real <lower=0> sigma; // change between seasons 1 and 2, scale parameter
    
}
transformed parameters{
  vector [S] phi;
  matrix[H+1,S] beta; // habitat sensitivity by species 
  for (i in 1:H+1){
    for (j in 1:S){
    beta[i,j] = mu[i]+tau[i]*beta_tilde[i,j];
  //implies habitat ~ normal(mu_habitat, sig_habitat
    }
  }
  for (j in 1:S){
        phi[j] = rho + sigma*phi_tilde[j];

  }
  // implies p ~ logistic(mu_p, sig_p)
}

model {
  // Priors
  // priors on beta:
  mu[1] ~ normal(-3,1); //water; based on mu1[1] from model with only year 2 
  mu[2] ~ normal(-3,2); //buildings
  mu[3] ~ normal(-3,2); //paved
  mu[4] ~ normal(-2.3,1); //barren
  mu[5] ~ normal(-2,2); // grass-herb
  mu[6] ~ normal(-2,2); // coniferous 
  mu[7] ~ normal(-2,2); // coniferous 
  mu[8] ~ normal(-3,2);
  
  tau[1] ~ normal(4,2);
  tau[2] ~ normal(4,2);
  tau[3] ~ normal(4,2);
  tau[4] ~ normal(7.3,1);
  tau[5] ~ normal(4,2);
  tau[6] ~ normal(4,2);
  tau[7] ~ normal(4,2);
  tau[8] ~ normal(4,2);

  //priors on p:
  psi ~ beta(2,3); // following Gelmen et al 2013, ch. 5, and thinner for rhat 
  theta ~ gamma(9,.5); // following Kruschke & Vanpaemel, 2015 
  //rho1 ~ normal(0,10);
  //sigma1 ~ normal(0,10);
  rho ~ normal(0,4);
  sigma ~ normal(.74,0.1);
  
  //phi ~ normal(rho, sigma);
  //phi2 ~ normal(rho2, sigma2);

  p ~ beta(theta*psi, theta*(1-psi));
  p[6] ~ normal(0.514, 0.01);
  p[7] ~ normal(0.577, 0.01); // based on p[7] from model with only year 2
  p[40] ~ normal(0.479, 0.01);
  p[43]~ normal(0.254, 0.01); 
  p[54] ~ normal(0.366, 0.01);
  to_vector(beta_tilde) ~ std_normal(); //implies habitat ~ normal(mu, tau)
  to_vector(phi_tilde) ~ std_normal();


  // Likelihood
  for (i in 1:R) {
    for (s in 1:S){ 
      // season 2: 
      vector[K - max_y2[i,s] + 1] lp2; //it's the product, lambda*p that's calculated
        
      for (j in 1:(K - max_y2[i,s] + 1)){
           lp2[j] = poisson_log_lpmf(max_y2[i,s] + j - 1 |E2_opt[i,s,]*beta[,s]*phi[s]) 
               + binomial_lpmf(y2[i,,s] | max_y2[i,s] + j - 1, p[s]); // implicitly sums across T (y[i] is vectorized)
      }
      target += log_sum_exp(lp2);
     // season 1
      vector[K - max_y1[i,s] + 1] lp1;

            for (j in 1:(K - max_y1[i,s] + 1)){
           lp1[j] = poisson_log_lpmf(max_y1[i,s] + j - 1 |E1_opt[i,s,]*beta[,s])
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
      N1[i,s] = poisson_log_rng(E1_opt[i,s,]*beta[,s]);
      N2[i,s] = poisson_log_rng(E2_opt[i,s,]*beta[,s]*phi[s]);

      //PPC[i,s] = binomial_rng (N[i,s], p[s] ) ;
    }
  }
}
