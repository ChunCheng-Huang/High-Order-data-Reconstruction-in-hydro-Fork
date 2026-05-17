///////////////////////////////////////////////////////////////////
/// 
/// 2-dimensional Advection Equation
/// 
///////////////////////////////////////////////////////////////////
/// Notes :  
/// 05/17 | Done | Setting Grid and Initial Value , RK4 Solver and
///              | Advection operator. If need other Method, just 
///              | change the Operator.
/// 05/18 | Work | NSE 
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

int     Nx = 256 , Ny = 256 , xmax = 10 , ymax = 10;
double  dx = 2*float(xmax)/Nx , dy = 2*float(ymax)/Ny;
string  Dir="Data" ,prob="Adv";

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
struct ConservationState {
    double rho ;
    double u, v, w ;

    ConservationState operator+(const ConservationState& other) const {
        return {rho + other.rho,
                u   + other.u  ,
                v   + other.v  ,
                w   + other.w};
    };

    ConservationState operator*(double scalar) const {
        return {rho * scalar,
                u   * scalar,
                v   * scalar, 
                w   * scalar};
    }
};

using RhsOperator = function<vector<vector<ConservationState>>(const vector<vector<ConservationState>>&)>;

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////  Initial  /////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

void Initial_Value( vector<vector<ConservationState>>& u0 ,vector<double>& x , vector<double>& y){
    // V1 - Testing struct
    
    #   pragma omp parallel for collapse(2)
    for (int i=0 ; i<Nx ; i++){
        for (int j=0 ; j<Ny ; j++){

            // Initail condiction
            double r = (x[i]*x[i] + y[j]*y[j]);
            u0[i][j].rho = exp(-r/4.0);
            u0[i][j].u = 1.0;
            u0[i][j].v = 0.0;
            u0[i][j].w = 0.0;
        }    
    }

    string Fname = "initial.csv";

    ofstream outFile("../" + Dir + "/" + prob + "_" +Fname);
    // The parameters
    outFile << "x,y,rho,u,v,w\n";

    for (int i=0 ; i < Nx ; i++){
        for (int j=0 ; j < Ny ; j++){
            outFile << x[i]       << " , " << y[j]       << " , " << u0[i][j].rho << " , ";
            outFile << u0[i][j].u << " , " << u0[i][j].v << " , " << u0[i][j].w   << '\n';
        }
    }

    outFile.close();

}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////  PDE Operator  ////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
vector<vector<ConservationState>> Pure_Advection_Operator(const vector<vector<ConservationState>>& U) {

    vector<vector<ConservationState>> dUdt(Nx, vector<ConservationState>(Ny, {0.0, 0.0, 0.0, 0.0}));
    
    // Background wind
    double constant_u = 1.0; 
    double constant_v = 1.0; 

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < Nx; i++) {
        for (int j = 1; j < Ny; j++) {

            double dphi_dx = (U[i][j].rho - U[i-1][j].rho) / dx;
            double dphi_dy = (U[i][j].rho - U[i][j-1].rho) / dy;

            dUdt[i][j].rho = -(constant_u * dphi_dx + constant_v * dphi_dy);
            
            dUdt[i][j].u = 0.0;
            dUdt[i][j].v = 0.0;
            dUdt[i][j].w = 0.0;
        }
    }
    return dUdt;
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////  RK4 Solver  //////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
vector<vector<ConservationState>> RK4_Step(
    const vector<vector<ConservationState>>& Un, 
    double dt, 
    RhsOperator calc_RHS) 
{
    // Stage 1
    vector<vector<ConservationState>> k1 = calc_RHS(Un);

    // Stage 2
    vector<vector<ConservationState>> U_temp(Nx, vector<ConservationState>(Ny));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            U_temp[i][j] = Un[i][j] + k1[i][j] * (0.5 * dt);
        }
    }
    vector<vector<ConservationState>> k2 = calc_RHS(U_temp);

    // Stage 3
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            U_temp[i][j] = Un[i][j] + k2[i][j] * (0.5 * dt);
        }
    }
    vector<vector<ConservationState>> k3 = calc_RHS(U_temp);

    // Stage 4
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            U_temp[i][j] = Un[i][j] + k3[i][j] * dt;
        }
    }
    vector<vector<ConservationState>> k4 = calc_RHS(U_temp);
    // Final Integration
    vector<vector<ConservationState>> Unext(Nx, vector<ConservationState>(Ny));
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
    vector< vector <ConservationState> > u0(Nx, vector <ConservationState> (Ny));

    // setting grid
    Grid_initail(x,y);

    // initial value
    Initial_Value(u0, x, y);

    // save data to csv or something else
    string Fname = "Final.csv";
    ofstream outFile("../" + Dir + "/" + prob + "_" +Fname);
    outFile << "time,x,y,rho,u,v,w\n";

    // RK4 or RK-high
    double dt = 0.01;
    int total_step  = 100;

    // Evolution
    cout << "Starting Time Evolution ... " << endl;
    for (int step=1 ; step <= total_step ; step++){

        // Pure Advection
        u0 = RK4_Step(u0 , dt , Pure_Advection_Operator);

        // NSE 
        // u0 = RK4_Step(u0 , dt , NSE_Operator);

        if (step % 20 == 0) {
            cout << "Step: " << step << " | Center density rho = " << u0[Nx/2][Ny/2].rho << endl;

            for (int i=0 ; i < Nx ; i++){
                for (int j=0 ; j < Ny ; j++){
                    outFile << step       << " , ";
                    outFile << x[i]       << " , " << y[j]       << " , " << u0[i][j].rho << " , ";
                    outFile << u0[i][j].u << " , " << u0[i][j].v << " , " << u0[i][j].w   << '\n';
                }
            }
        }
    }


    outFile.close();

    cout << "End time evolution "<< endl;
}

