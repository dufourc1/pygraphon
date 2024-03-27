import cppyy
cppyy.include('cppyy_backend/include/RcppCommon.h')
cppyy.include('cppyy_backend/include/Rcpp.h')
cppyy.include('cppyy_backend/include/RcppArmadillo.h')

cppyy.cppdef('''
    using namespace Rcpp;
    using namespace arma;
    NumericVector nn_TSP(NumericMatrix f,int p, int n,int init)
    {
    NumericVector state(n,0.0);
    state[init-1] = 1;
    NumericVector order(n,0.0); 
    order[0] = init;

    int next;
    float dist;
    float max_dist = pow(10.0,15);
    float temp1;

    for(int i = 1; i < n; i++)
    {
    next = 1;
    dist = max_dist;
    int k = order[i-1];

    for(int j = 1; j<(n+1);j++)
    {
    if( state[j-1] < .5)
    {
    temp1 = 0.0;
    for(int s = 0; s<p;s++)
    {
    temp1 = temp1 +  (f(k-1,s)  - f(j-1,s))*(f(k-1,s)  - f(j-1,s));
    }
    if( temp1 < dist)
    {
    dist = temp1;
    next = j;
    }
    }
    }
    order[i] = next;  
    state[next-1] = 1; 
    }
    return order;  
    }
''')

def call_nn_TSP(f, p, n, init):
    f = cppyy.gbl.Rcpp::NumericMatrix(f)
    n = cppyy.gbl.int(n)
    p = cppyy.gbl.int(p)
    init = cppyy.gbl.int(init)
    return cppyy.gbl.nn_TSP(f, p, n, init)