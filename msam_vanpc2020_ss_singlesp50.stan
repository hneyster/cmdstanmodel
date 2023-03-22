//  Multi-species Abundance Model (MSAM) to model the abundance of birds in Vancovuer in 2020
// using data replicated from melles (2000)
// Written by Harold Eyster 
// revised, added beta dist on p. 
// added spatial scale selection 
// functions {
//   row_vector scale_select(vector zeta, matrix E) {
//     //int C = cols(E);
//     //real zeta_trans = zeta*(C-1)+ 1; //translating zeta  from 0,1 to 1,30 
//     array index[rows(E)] int  = cols(E);
//     vector [rows(E)] interp_index;
//     for (i in zeta){
//     // If zeta is already above the maximum index then
//     // we can skip the interpolation
//     if (index < zeta_trans) {
//       return E[index,:];
//     }
//     
//     while (index > zeta_trans) { 
//       index -= 1;
//     }
//     
//     real prop_scales = zeta_trans - index;
//     }
//     
//     // Assumes same value of zeta for every row
//     return E[index,:] * (1 - prop_scales) + E[index + 1,:] * prop_scales;
//   }
// }
data {
  int<lower=0> R;       // Number of sites in season 1
  int<lower=0> J;       // Number of temporal replicates in season 2
  int<lower=0> S;       // Number of species 
  int<lower=0> H;       // number of habitat types 
  int<lower=0> C;       // number of scales 
  int<lower=0> y2[R,J, S]; // Counts in season 1
  int<lower=0> K;       // Upper bound of population size
  array [R] matrix [C,H] E2;       // environmental covariates in season 2 
}

transformed data {
  int<lower=0> max_y2[R]; // the the max number of each sp observed at each spatial replication.
  for (i in 1:R) {
      max_y2[i] = max(y2[i,,1]); 
  }
 array[H] int zeta = { 1, 1, 1, 1, 1, 1, 1 };
 matrix[R,H] Emod; 
 for (i in 1:R){
   Emod[i,:] = to_row_vector(diagonal(E2[i,zeta,:]));
 }
 

}
parameters {
    //vector <lower=0, upper = 1>[H] zeta1;
    // vector <lower=0, upper = 1>[H] zeta2;
    real <lower=0, upper=1> psi;
    real<lower=1.0/C,upper=1> kappa;
    real <lower=0> theta;
    real <lower=0> omega;// 
    vector [H] beta; // centered habitat weights 
    real <lower=0,upper=1> p;  // centered detection variation by species 
    simplex[2] rho;
    //vector [S] phi2; 
    //real rho2;
    //real <lower=0> sigma2; 
    
}
model {
  // Priors
  // priors on beta:
  beta[1] ~ normal(-2,2); //water 
  beta[2] ~ normal(-3,2); //buildings
  beta[3] ~ normal(-3,2); //paved
  beta[4] ~ normal(-2,2); //barren
  beta[5] ~ normal(-2,2); // grass-herb
  beta[6] ~ normal(-2,2); // coniferous 

  // tau ~ normal(2,2);
  //priors on p:
  //psi ~ beta(2,3); // following Gelmen et al 2013, ch. 5, and thinner for rhat 
  //theta ~ gamma(9,.5); // following Kruschke & Vanpaemel, 2015 
  //rho1 ~ normal(0,10);
  //sigma1 ~ normal(0,10);
  //omega ~ gamma(2,4);
  //kappa ~ beta(1,1);
  
  //phi ~ normal(rho, sigma);
  //phi2 ~ normal(rho2, sigma2);
  //zeta1 ~ normal(1,5);
  //zeta2 ~ normal(30,5);
  p ~ beta(2,2); 
  // Likelihood
  for (i in 1:R) {
      // season 2: 
      vector[K - max_y2[i] + 1] lp2; //it's the product, lambda*p that's calculated
        
      for (j in 1:(K - max_y2[i] + 1)){
           lp2[j] = poisson_log_lpmf(max_y2[i] + j - 1 |Emod[i,]*beta) 
               + binomial_lpmf(y2[i,,1] | max_y2[i] + j - 1, p); // implicitly sums across T (y[i] is vectorized)
      }
      target += log_sum_exp(lp2);
    }
}

generated quantities{
  int N2[R];
 // int PPC[R,S];

  for (i in 1:R){
      N2[i] = poisson_log_rng(Emod[i,]*beta);

      //PPC[i,s] = binomial_rng (N[i,s], p[s] ) ;
  }
}
