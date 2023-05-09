//  Multi-species Abundance Model (MSAM) to model the abundance of birds in Vancovuer in 2020
// using data replicated from melles (2000)
// Written by Harold Eyster 
// revised, added beta dist on p. 
// added 1997 bird observations 
// added phi, but as a multiplyer 
// changing phi to an summation, since exp(-3*-3) = positive. 
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
    real <lower=0> theta; // 
    vector[H] mu;
    vector<lower=0>[H] tau;
    matrix [H,S] beta_tilde; // centered habitat weights 
    vector <lower=0,upper=1> [S] p;  // centered detection variation by species 
    vector [S] phi_tilde;
    real rho;
    real <lower=0> sigma; 
    //vector [S] phi2; 
    //real rho2;
    //real <lower=0> sigma2; 
    
}
transformed parameters{
  vector [S] phi;
  matrix[H,S] beta; // habitat sensitivity by species 
  for (i in 1:H){
    for (j in 1:S){
    beta[i,j] = mu[i]+tau[i]*beta_tilde[i,j];
    phi[j] = rho + sigma*phi_tilde[j];
  //implies habitat ~ normal(mu_habitat, sig_habitat
    }
  }
  // implies p ~ logistic(mu_p, sig_p)
}

model {
  // Priors
  // priors on beta:
  mu[1] ~ normal(-3,1); //water; based on mu1[1] from model with only year 2 
  mu[2] ~ normal(-3,2); //buildings
  mu[3] ~ normal(-3,2); //paved
  mu[4] ~ normal(-2,2); //barren
  mu[5] ~ normal(-2,2); // grass-herb
  mu[6] ~ normal(-2,2); // coniferous 
  mu[7] ~ normal(-2,2); // coniferous 
  
  tau ~ normal(4,2);

  
  //priors on p:
  psi ~ beta(2,3); // following Gelmen et al 2013, ch. 5, and thinner for rhat 
  theta ~ gamma(9,.5); // following Kruschke & Vanpaemel, 2015 
  //rho1 ~ normal(0,10);
  //sigma1 ~ normal(0,10);
  rho ~ normal(0,4);
  sigma ~ normal(1,2);
  
  //phi ~ normal(rho, sigma);
  //phi2 ~ normal(rho2, sigma2);

  p ~ beta(theta*psi, theta*(1-psi));
  p[7] ~ normal(0.57, 0.05); // based on p[7] from model with only year 2
  to_vector(beta_tilde) ~ std_normal(); //implies habitat ~ normal(mu, tau)
  to_vector(phi_tilde) ~ std_normal();

  // Likelihood
  for (i in 1:R) {
    for (s in 1:S){ 
      // season 2: 
      vector[K - max_y2[i,s] + 1] lp2; //it's the product, lambda*p that's calculated
        
      for (j in 1:(K - max_y2[i,s] + 1)){
           lp2[j] = poisson_log_lpmf(max_y2[i,s] + j - 1 |E2[i,]*beta[,s] + phi[s]) 
               + binomial_lpmf(y2[i,,s] | max_y2[i,s] + j - 1, p[s]); // implicitly sums across T (y[i] is vectorized)
      }
      target += log_sum_exp(lp2);
      // season 1 
      vector[K - max_y1[i,s] + 1] lp1;
      
            for (j in 1:(K - max_y1[i,s] + 1)){
           lp1[j] = poisson_log_lpmf(max_y1[i,s] + j - 1 |E1[i,]*beta[,s]) 
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
      N1[i,s] = poisson_log_rng(E1[i,]*beta[,s]);
      N2[i,s] = poisson_log_rng(E2[i,]*beta[,s] + phi[s]);

      //PPC[i,s] = binomial_rng (N[i,s], p[s] ) ;
    }
  }
}
