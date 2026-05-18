///////////////////////////////////////////////////////////////////
/// 
/// 2-dimensional Navier-Stokes Equation
/// 
///================================================================
/// Notes :  
/// 05/17 | Done | Setting Grid and Initial Value , RK4 Solver and
///              | Advection operator. If need other Method, just 
///              | change the Operator.
/// 05/18 | Work | NSE 
///
///================================================================
/// Reference : 
///     
/// 
///////////////////////////////////////////////////////////////////

#include <iostream>
#include <omp.h>
#include <vector>
#include <cmath>
#include <fstream>
#include <functional>

using namespace std;

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////  GLOBAL  //////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
// Change here to setting in another txt file ???

const int     Nx = 256   , xmax = 10, Ny = 256   , ymax = 10;
const double  dx = 2*float(xmax)/Nx , dy = 2*float(ymax)/Ny ;
const string  Dir="Data" , prob = "NSE";

// Physics constant
const double cs = 1.0; //

////////////////////////////////////////////////////////////////////////////
/////////////////////////////  Grid  ///////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
void Grid_initail(vector<double>& x , vector<double>& y ){
    
    #   pragma omp parallel for
    for (int i=0 ; i<Nx ; i++){x[i]=-xmax + (i+0.5)*dx;}    
    #   pragma omp parallel for
    for (int i=0 ; i<Ny ; i++){y[i]=-ymax + (i+0.5)*dy;}    
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////  Data Struct  /////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
struct ConsState {
    double rho;  // Dsensity
    double mu ;  // x momenta rho * u
    double mv ;  // y momenta rho * v
    double mw ;  // z momenta rho * w

    ConsState operator+(const ConsState& other) const {
        return {rho + other.rho ,
                mu  + other.mu  ,
                mv  + other.mv  ,
                mw  + other.mw };
    };

    ConsState operator*(double scalar) const {
        return {rho * scalar  ,
                mu  * scalar  ,
                mv  * scalar  , 
                mw  * scalar };
    }
};

using RhsOperator = function<vector<vector<ConsState>>(const vector<vector<ConsState>>&)>;

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////  Initial  /////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

void Initial_Value( vector<vector<ConsState>>& u0 ,vector<double>& x , vector<double>& y){
    // V1 - Testing struct
    
    #   pragma omp parallel for collapse(2)
    for (int i=0 ; i<Nx ; i++){
        for (int j=0 ; j<Ny ; j++){

            // Initail condiction
            double r = (x[i]*x[i] + y[j]*y[j]);
            u0[i][j].rho = exp(-r/4.0);
            u0[i][j].mu  = -1e5*y[j]/sqrt(r)*sin(sqrt(r) * M_PI) * u0[i][j].rho;
            u0[i][j].mv  =  1e5*x[i]/sqrt(r)*sin(sqrt(r) * M_PI) * u0[i][j].rho;
            u0[i][j].mw  = 0.0 * u0[i][j].rho;
        }    
    }

    string Fname = "initial.csv";

    ofstream outFile("../" + Dir + "/"+ prob + "_" + Fname);
    // The parameters
    outFile << "x,y,rho,mu,mv,mw\n";

    for (int i=0 ; i < Nx ; i++){
        for (int j=0 ; j < Ny ; j++){
            outFile << x[i]        << " , " << y[j]        << " , " << u0[i][j].rho  << " , ";
            outFile << u0[i][j].mu << " , " << u0[i][j].mv << " , " << u0[i][j].mw   << '\n';
        }
    }
    outFile.close();
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////  PDE Operator  ////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
vector<vector<ConsState>> NSE_Operator(const vector<vector<ConsState>>& U) {

    vector<vector<ConsState>> dUdt(Nx, vector<ConsState>(Ny, {0.0, 0.0, 0.0, 0.0}));

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < Nx-1 ; i++) {
        for (int j = 1; j < Ny-1 ; j++) {

            double u   = U[i][j].mu/U[i][j].rho;
            double v   = U[i][j].mv/U[i][j].rho;
            // double P = cs * cs * U[i][j].rho;
            
            // Continuity Equation
            double drho_dx = (U[i+1][j].rho - U[i-1][j].rho ) / (2.0 * dx); 
            double drho_dy = (U[i][j+1].rho - U[i][j-1].rho ) / (2.0 * dy); 

            // Momenta Equation
            // X
            double dmu_dx = (U[i+1][j].mu  - U[i-1][j].mu ) /  (2.0 * dx) ;
            double dmu_dy = (U[i][j+1].mu  - U[i][j-1].mu ) /  (2.0 * dy) ;
            double du_dx  = (U[i+1][j].mu / U[i+1][j].rho  - U[i-1][j].mu / U[i-1][j].rho) /  (2.0 * dx) ;
            double dv_dy  = (U[i][j+1].mv / U[i][j+1].rho  - U[i][j-1].mv / U[i][j-1].rho) /  (2.0 * dy) ;
            double dP_dx  = cs * cs * ( U[i+1][j].rho - U[i-1][j].rho) / (2.0 * dx);

            // Y
            double dmv_dx = (U[i+1][j].mv  - U[i-1][j].mv ) /  (2.0 * dx) ;
            double dmv_dy = (U[i][j+1].mv  - U[i][j-1].mv ) / (2.0 * dy) ;
            double dP_dy  = cs * cs * ( U[i+1][j].rho - U[i-1][j].rho) / (2.0 * dy);
            
            // Time differential
            dUdt[i][j].rho  = -drho_dx-drho_dy;
            dUdt[i][j].mu   = -( u*dmu_dx + v*dmu_dy + dP_dx + U[i][j].mu * (du_dx+dv_dy));
            dUdt[i][j].mv   = -( u*dmv_dx + v*dmv_dy + dP_dy + U[i][j].mv * (du_dx+dv_dy));;
            dUdt[i][j].mw   = 0.0;

        }
    }
    return dUdt;
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////  RK4 Solver  //////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
vector<vector<ConsState>> RK4_Step(
    const vector<vector<ConsState>>& Un, 
    double dt, 
    RhsOperator calc_RHS) 
{
    // Stage 1
    vector<vector<ConsState>> k1 = calc_RHS(Un);

    // Stage 2
    vector<vector<ConsState>> U_temp(Nx, vector<ConsState>(Ny));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            U_temp[i][j] = Un[i][j] + k1[i][j] * (0.5 * dt);
        }
    }
    vector<vector<ConsState>> k2 = calc_RHS(U_temp);

    // Stage 3
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            U_temp[i][j] = Un[i][j] + k2[i][j] * (0.5 * dt);
        }
    }
    vector<vector<ConsState>> k3 = calc_RHS(U_temp);

    // Stage 4
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            U_temp[i][j] = Un[i][j] + k3[i][j] * dt;
        }
    }
    vector<vector<ConsState>> k4 = calc_RHS(U_temp);
    // Final Integration
    vector<vector<ConsState>> Unext(Nx, vector<ConsState>(Ny));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            Unext[i][j] = Un[i][j] + (k1[i][j] + k2[i][j] * 2.0 + k3[i][j] * 2.0 + k4[i][j]) * (dt / 6.0);
        }
    }
    return Unext;
}

///////////////////////////////////////////////////////////////////

int main(){
    vector<double> x(Nx) , y(Ny);
    vector< vector <ConsState> > u0(Nx, vector <ConsState> (Ny));

    // setting grid
    Grid_initail(x,y);

    // initial value
    Initial_Value(u0, x, y);

    // save data to csv or something else
    string Fname = "Final.csv";
    ofstream outFile("../" + Dir + "/"+ prob + "_" +Fname);
    outFile << "time,x,y,rho,mu,mv,mw\n";

    // RK4 or RK-high
    double dt = 0.01;
    int total_step  = 1000;

    // Evolution
    cout << "Starting Time Evolution ... " << endl;
    for (int step=1 ; step <= total_step ; step++){

        // NSE 
        u0 = RK4_Step(u0 , dt , NSE_Operator);

        if (step % 20 == 0) {
            cout << "Step: " << step << " | Center density rho = " << u0[Nx/2][Ny/2].rho << endl;

            for (int i=0 ; i < Nx ; i++){
                for (int j=0 ; j < Ny ; j++){
                    outFile << step        << " , ";
                    outFile << x[i]        << " , " << y[j]        << " , " << u0[i][j].rho  << " , ";
                    outFile << u0[i][j].mu << " , " << u0[i][j].mv << " , " << u0[i][j].mw   << '\n';
                }
            }
        }
    }


    outFile.close();

    cout << "End time evolution "<< endl;
}

