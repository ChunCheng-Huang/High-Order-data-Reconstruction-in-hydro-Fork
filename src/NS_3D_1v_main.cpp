///////////////////////////////////////////////////////////////////
/// 
/// 2-dimensional Navier-Stokes Equation
/// 
///================================================================
/// Notes :  
/// 05/17 | Done | First try 3D NSE simulation.
///              | 
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
#include <cstdio>
#include <filesystem>

using namespace std;
// namespace fs = std::filesystem;

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////  GLOBAL  //////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
// Change here to setting in another txt file ???

const int     Nx = 32 ,   xmax = 10 , Ny = 32   , ymax = 10 , Nz = 32   , zmax = 10;
const double  dx = 2*float(xmax)/Nx , dy = 2*float(ymax)/Ny , dz = 2*float(zmax)/Nz ;
const string  Dir="Data" , prob = "NSE_3D_test1";
const double  inv_dx2 = 1/(dx*dx) , inv_dy2 = 1/(dy*dy) ,inv_dx = 1/(2*dx) , inv_dy = 1/(2*dy);

// Physics constant
const double cs  = 1.0;  // Adiabatic 
const int Estep  = 100;  // Each step
const double eta = 0.1;  // Viscosity

////////////////////////////////////////////////////////////////////////////
/////////////////////////////  Grid  ///////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
void Grid_initail(vector<double>& x , vector<double>& y , vector<double>& z ){
    
    #   pragma omp parallel for
    for (int i=0 ; i<Nx ; i++){x[i]=-xmax + (i+0.5)*dx;}    
    #   pragma omp parallel for
    for (int i=0 ; i<Ny ; i++){y[i]=-ymax + (i+0.5)*dy;}    
    #   pragma omp parallel for
    for (int i=0 ; i<Nz ; i++){z[i]=-zmax + (i+0.5)*dz;}    
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

inline double S_rho(double rho) { return (rho < 1e-10) ? 1e-10 : rho; }

inline int idx(int i, int j , int k)    { return i * Ny *Nz + j * Nz + k; }

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////  Initial  /////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

void Initial_Value( vector<ConsState>& u0 ,vector<double>& x , vector<double>& y , vector<double>& z){
    // V1 - Testing struct
    
    #   pragma omp parallel for collapse(3)
    for (int i=0 ; i<Nx ; i++){
        for (int j=0 ; j<Ny ; j++){
            for (int k=0 ; k<Nz ; k++){
                // Initail condiction
                double r = (x[i]*x[i] + y[j]*y[j] + z[k]*z[k]);
                u0[idx(i, j, k)].rho = exp(-r/4.0);
                u0[idx(i, j, k)].mu  = -1e0*y[j]/sqrt(r)*sin(sqrt(r) * M_PI) * u0[idx(i, j, k)].rho;
                u0[idx(i, j, k)].mv  =  1e0*x[i]/sqrt(r)*sin(sqrt(r) * M_PI) * u0[idx(i, j, k)].rho;
                u0[idx(i, j, k)].mw  = 0.0 * u0[idx(i, j, k)].rho;
            }
        }    
    }

    string Fname = "initial.csv";
    ofstream outFile("../" + Dir + "/"+ prob + "/" + prob + "_" + Fname);
    cout << "../" + Dir + "/"+ prob + "/" + prob + "_" + Fname << endl;

    // The parameters
    outFile << "x,y,rho,mu,mv,mw\n";

    for (int i=0 ; i < Nx ; i++){
        for (int j=0 ; j < Ny ; j++){
            for (int k=0 ; k<Nz ; k++){
                outFile << x[i]                << " , " << y[j]                << " , " << u0[idx(i, j, k)].rho  << " , ";
                outFile << u0[idx(i, j, k)].mu << " , " << u0[idx(i, j, k)].mv << " , " << u0[idx(i, j, k)].mw   << '\n';
            }
        }
    }
    outFile.close();
}

// ///////////////////////////////////////////////////////////////////
// void Apply_Boundary(vector<ConsState>& U) {
//     for (int i = 0; i < Nx; i++) {
//         U[idx(i, 0)] = U[idx(i, Ny - 2)];
//         U[idx(i, Ny - 1)] = U[idx(i, 1)];
//     }
//     for (int j = 0; j < Ny; j++) {
//         U[idx(0, j)] = U[idx(Nx - 2, j)];
//         U[idx(Nx - 1, j)] = U[idx(1, j)];
//     }
// }
// /////////////////////////////////////////////////////////////////////////////
// /////////////////////////////  PDE Operator  ////////////////////////////////
// /////////////////////////////////////////////////////////////////////////////
// void NSE_Operator(const vector<ConsState>& U, vector<ConsState>& dUdt) {

//     #pragma omp parallel for collapse(2)
//     for (int i = 1; i < Nx - 1; i++) {
//         for (int j = 1; j < Ny - 1; j++) {
//             double rho = S_rho(U[idx(i, j)].rho);
//             if (U[idx(i, j)].rho < 1e-10) rho=1e-10;

//             double u   = U[idx(i, j)].mu/rho;
//             double v   = U[idx(i, j)].mv/rho;
            
//             // Continuity Equation
//             double drho_dx = (U[idx(i+1, j)].rho - U[idx(i-1, j)].rho ) * inv_dx; 
//             double drho_dy = (U[idx(i, j+1)].rho - U[idx(i, j-1)].rho ) * inv_dy; 

//             // Momenta Equation
//             // X
//             double dmu_dx  = (U[idx(i+1, j)].mu - U[idx(i-1, j)].mu ) * inv_dx ;
//             double dFmu_XL = (U[idx(i-1, j)].mu * U[idx(i-1, j)].mu   / S_rho(U[idx(i-1, j)].rho)) + cs * cs * S_rho(U[idx(i-1, j)].rho);
//             double dFmu_XR = (U[idx(i+1, j)].mu * U[idx(i+1, j)].mu   / S_rho(U[idx(i+1, j)].rho)) + cs * cs * S_rho(U[idx(i+1, j)].rho);
//             double dFmu_YD = (U[idx(i, j-1)].mu * U[idx(i, j-1)].mv   / S_rho(U[idx(i, j-1)].rho)) ;
//             double dFmu_YU = (U[idx(i, j+1)].mu * U[idx(i, j+1)].mv   / S_rho(U[idx(i, j+1)].rho)) ;

//             double du2dx   = (U[idx(i+1, j)].mu - U[idx(i ,  j)].mu * 2 + U[idx(i-1, j)].mu) * inv_dx2;
//             double du2dy   = (U[idx(i, j+1)].mu - U[idx(i ,  j)].mu * 2 + U[idx(i, j-1)].mu) * inv_dy2;

//             // Y
//             double dmv_dy  = (U[idx(i, j+1)].mv - U[idx(i, j-1)].mv ) * inv_dy ;
//             double dFmv_YD = (U[idx(i, j-1)].mv * U[idx(i, j-1)].mv   / S_rho(U[idx(i, j-1)].rho)) + cs * cs * S_rho(U[idx(i, j-1)].rho);
//             double dFmv_YU = (U[idx(i, j+1)].mv * U[idx(i, j+1)].mv   / S_rho(U[idx(i, j+1)].rho)) + cs * cs * S_rho(U[idx(i, j+1)].rho);
//             double dFmv_XL = (U[idx(i-1, j)].mv * U[idx(i-1, j)].mu   / S_rho(U[idx(i-1, j)].rho)) ;
//             double dFmv_XR = (U[idx(i+1, j)].mv * U[idx(i+1, j)].mu   / S_rho(U[idx(i+1, j)].rho)) ;

//             double dv2dx   = (U[idx(i+1, j)].mv - U[idx(i ,  j)].mv * 2 + U[idx(i-1, j)].mv) * inv_dx2;
//             double dv2dy   = (U[idx(i, j+1)].mv - U[idx(i ,  j)].mv * 2 + U[idx(i, j-1)].mv) * inv_dy2;

//             // Time differential
//             dUdt[idx(i, j)].rho  = - dmu_dx - dmv_dy;
//             dUdt[idx(i, j)].mu   = -(dFmu_XR - dFmu_XL) * inv_dx - (dFmu_YU - dFmu_YD) * inv_dy + eta * (du2dx + du2dy);
//             dUdt[idx(i, j)].mv   = -(dFmv_XR - dFmv_XL) * inv_dx - (dFmv_YU - dFmv_YD) * inv_dy + eta * (dv2dx + dv2dy);
//             dUdt[idx(i, j)].mw   = 0.0;

//         }
//     }
//     Apply_Boundary(dUdt);

// }

// /////////////////////////////////////////////////////////////////////////////
// /////////////////////////////  RK4 Solver  //////////////////////////////////
// /////////////////////////////////////////////////////////////////////////////
// void RK4_Step(
//     const vector<ConsState>& Un, 
//     vector<ConsState>& Unext,
//     vector<ConsState>& k1, vector<ConsState>& k2, vector<ConsState>& k3, vector<ConsState>& k4,
//     vector<ConsState>& U_temp,
//     double dt) 
// {
//     // Stage 1
//     NSE_Operator(Un, k1);

//     // Stage 2
//     #pragma omp parallel for
//     for (int i = 0; i < Nx * Ny; i++) U_temp[i] = Un[i] + k1[i] * (0.5 * dt);
//     Apply_Boundary(U_temp);
//     NSE_Operator(U_temp, k2);

//     // Stage 3
//     #pragma omp parallel for
//     for (int i = 0; i < Nx * Ny; i++) U_temp[i] = Un[i] + k2[i] * (0.5 * dt);
//     Apply_Boundary(U_temp);
//     NSE_Operator(U_temp, k3);

//     // Stage 4
//     #pragma omp parallel for
//     for (int i = 0; i < Nx * Ny; i++) U_temp[i] = Un[i] + k3[i] * dt;
//     Apply_Boundary(U_temp);
//     NSE_Operator(U_temp, k4);

//     // Final Integration
//     #pragma omp parallel for
//     for (int i = 0; i < Nx * Ny; i++) {
//         Unext[i] = Un[i] + (k1[i] + k2[i] * 2.0 + k3[i] * 2.0 + k4[i]) * (dt / 6.0);
//     }
// }


///////////////////////////////////////////////////////////////////

int main(){
    filesystem::path dir_path = "../Data/" + prob ;  
    if (!filesystem::exists(dir_path) ) {
        if(filesystem::create_directory(dir_path)){cout << "Success build " << dir_path << endl;}
    }

    vector<double> x(Nx) , y(Ny) ,z(Nz);
    vector<ConsState> u0(Nx * Ny * Nz), u_next(Nx * Ny * Nz);
    // vector<ConsState> k1(Nx * Ny), k2(Nx * Ny), k3(Nx * Ny), k4(Nx * Ny), U_temp(Nx * Ny);

    // setting grid
    Grid_initail(x,y,z);

    // // initial value
    Initial_Value(u0, x, y, z);

    // // RK4 or RK-high
    // double dt = 0.0001;
    // int total_step  = 10000;

    // // Evolution
    // cout << "Starting Time Evolution ... " << endl;
    // for (int step=1 ; step <= total_step ; step++){

    //     // NSE 
    //     RK4_Step(u0, u_next, k1, k2, k3, k4, U_temp, dt);
    //     u0.swap(u_next);
    //     // Apply_Boundary(u0);

    //     // Check the center value
    //     if (step % Estep == 0) {
    //         char buffer[50];
    //         std::snprintf(buffer, sizeof(buffer), "%04d", step / Estep);
    //         std::string stepStr = buffer;

    //         cout << "Step: " << step << " | Center density rho = " << u0[idx(Nx/2,Ny/2)].rho << endl;

    //         string Fname = "Final.csv";
    //         // ofstream outFile("../" + Dir + "/"+ prob + "/" + prob + "_" + to_string(step/Estep) + Fname);
    //         ofstream outFile("../" + Dir + "/"+ prob + "/" + prob + "_" + stepStr + "_" + Fname);
    //         outFile << "time,x,y,rho,mu,mv,mw\n";
    //         for (int i=0 ; i < Nx ; i++){
    //             for (int j=0 ; j < Ny ; j++){
    //                 outFile << step            << " , ";
    //                 outFile << x[i]            << " , " << y[j]            << " , " << u0[idx(i,j)].rho  << " , ";
    //                 outFile << u0[idx(i,j)].mu << " , " << u0[idx(i,j)].mv << " , " << u0[idx(i,j)].mw   << '\n';
    //             }
    //         }
    //         outFile.close();
    //     }
    // }

    // cout << "End time evolution "<< endl;
}

