//  Multi-species Abundance Model (MSAM) to model the abundance of birds in Vancovuer in 2020
// using data replicated from melles (2000)
// Written by Harold Eyster 
// revised, added beta dist on p. 
// added 1997 bird observations 
// added phi, but as a multiplyer 
data {
  int<lower=0> R;       // Number of sites in season 1
  int<lower=0> J;       // Number of temporal replicates in season 2
  int<lower=0> S;       // Number of species 
  int<lower=0> H;       // number of habitat types 
  int<lower=0> y1[R, S]; // Counts in season 1
  int<lower=0> y2[R,J,S]; // counts in season 2
  int<lower=0> K;       // Upper bound of population size
  matrix [R,H] E1;      // environmental covariates in season 1
  matrix [R,H] E2;       // environmental covariates in season 2 
}

transformed data {
  int<lower=0> max_y2[R,S]; // the the max number of each sp observed at each spatial replication.
  int<lower=0> max_y1[R,S];
  max_y1 = y1; # no temporal replication, so it is the max 
  for (i in 1:R) {
    for (j in 1:S)
      max_y2[i,j] = max(y2[i,,j]); 
  }
}

parameters {
    real <lower=0, upper=1> psi;       // 
    real <lower=.1> theta; // 
    vector[H] mu;
    vector<lower=0>[H] tau;
    matrix [H,S] beta_tilde; // centered habitat weights 
    vector <lower=0,upper=1> [S] p;  // centered detection variation by species 
    vector [S] phi1; 
    real rho1;
    real <lower=0> sigma1; 
    vector [S] phi2; 
    real rho2;
    real <lower=0> sigma2; 
    
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
  
  tau ~ normal(0,2);
  //priors on p:
  psi ~ beta(1,1); // following Gelmen et al 2013, ch. 5 
  theta ~ pareto(.1,1.5); // following Gelmen et al 2013, ch. 5 
  rho1 ~ normal(0,10);
  sigma1 ~ normal(0,10);
  rho2 ~ normal(0,10);
  sigma2 ~ normal(0,10);
  
  phi1 ~ normal(rho1, sigma1);
  phi2 ~ normal(rho2, sigma2);

  p ~ beta(theta*psi, theta*(1-psi));

  to_vector(beta_tilde) ~ std_normal(); //implies habitat ~ normal(mu, tau)

  // Likelihood
  for (i in 1:R) {
    for (s in 1:S){ 
      // season 2: 
      vector[K - max_y2[i,s] + 1] lp2; //it's the product, lambda*p that's calculated
        
      for (j in 1:(K - max_y2[i,s] + 1)){
           lp2[j] = poisson_log_lpmf(max_y2[i,s] + j - 1 |E2[i,]*beta[,s]*phi2[s]) 
               + binomial_lpmf(y2[i,,s] | max_y2[i,s] + j - 1, p[s]); // implicitly sums across T (y[i] is vectorized)
      }
      target += log_sum_exp(lp2);
      // season 1 
      vector[K - max_y1[i,s] + 1] lp1;
      
            for (j in 1:(K - max_y1[i,s] + 1)){
           lp1[j] = poisson_log_lpmf(max_y1[i,s] + j - 1 |E1[i,]*beta[,s]*phi1[s]) 
               + binomial_lpmf(y1[i,s] | max_y1[i,s] + j - 1, p[s]); // implicitly sums across T (y[i] is vectorized)
      }
      target += log_sum_exp(lp1);
    }
  }
}

generated quantities{
  int N1[R,S];
  int N2[R,S];
 // int PPC[R,S];

  for (i in 1:R){
    for (s in 1:S){
      N1[i,s] = poisson_log_rng(E1[i,]*beta[,s]*phi1[s]);
      N2[i,s] = poisson_log_rng(E2[i,]*beta[,s]*phi2[s]);

      //PPC[i,s] = binomial_rng (N[i,s], p[s] ) ;
    }
  }
}
