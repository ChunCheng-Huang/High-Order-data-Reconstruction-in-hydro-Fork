#include <iostream>
#include <omp.h>
#include <vector>

using namespace std;

// 建議改為扁平化陣列，Nx * Ny 大小

const int     Nx = 256   , xmax = 10, Ny = 256   , ymax = 10;
const double  dx = 2*float(xmax)/Nx , dy = 2*float(ymax)/Ny ;
const string  Dir="Data" , prob = "NSE";

// Physics constant
const double cs = 1.0; //
struct ConsState {
     double rho, mu, mv, mw;
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

// 使用扁平化索引存取：i * Ny + j
inline int idx(int i, int j) { return i * Ny + j; }

// 傳入引用，且不需回傳值
void NSE_Operator(const vector<ConsState>& U, vector<ConsState>& dUdt) {
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < Nx - 1; i++) {
        for (int j = 1; j < Ny - 1; j++) {
            double u   = U[idx(i, j)].mu/U[idx(i, j)].rho;
            double v   = U[idx(i, j)].mv/U[idx(i, j)].rho;
          //   double P = cs * cs * U[idx(i, j)].rho;
            
            // Continuity Equation
            double drho_dx = (U[idx(i+1, j)].rho - U[idx(i-1, j)].rho ) / (2.0 * dx); 
            double drho_dy = (U[idx(i, j+1)].rho - U[idx(i, j-1)].rho ) / (2.0 * dy); 

          //   // Momenta Equation
          //   // X
            double dmu_dx = (U[idx(i+1, j)].mu  - U[idx(i-1, j)].mu ) /  (2.0 * dx) ;
            double dmu_dy = (U[idx(i, j+1)].mu  - U[idx(i, j-1)].mu ) /  (2.0 * dy) ;
            double du_dx  = (U[idx(i+1, j)].mu / U[idx(i+1, j)].rho  - U[idx(i-1, j)].mu / U[idx(i-1, j)].rho) /  (2.0 * dx) ;
            double dv_dy  = (U[idx(i, j+1)].mv / U[idx(i, j+1)].rho  - U[idx(i, j-1)].mv / U[idx(i, j-1)].rho) /  (2.0 * dy) ;
            double dP_dx  = cs * cs * ( U[idx(i+1, j)].rho - U[idx(i-1, j)].rho) / (2.0 * dx);

          //   // Y
            double dmv_dx = (U[idx(i+1, j)].mv  - U[idx(i-1, j)].mv ) / (2.0 * dx) ;
            double dmv_dy = (U[idx(i, j+1)].mv  - U[idx(i, j-1)].mv ) / (2.0 * dy) ;
            double dP_dy  = cs * cs * ( U[idx(i, j+1)].rho - U[idx(i, j-1)].rho) / (2.0 * dy);
            
            // Time differential
            dUdt[idx(i, j)].rho  = - dmu_dx - dmv_dy;
            dUdt[idx(i, j)].mu   = -( u*dmu_dx + v*dmu_dy + dP_dx + U[idx(i, j)].mu * (du_dx+dv_dy));
            dUdt[idx(i, j)].mv   = -( u*dmv_dx + v*dmv_dy + dP_dy + U[idx(i, j)].mv * (du_dx+dv_dy));;
            dUdt[idx(i, j)].mw   = 0.0;
          //   dUdt[idx(i, j)].rho = ... ; 
        }
    }
}

// 修改 RK4 為引用傳遞
void RK4_Step(
    const vector<ConsState>& Un, 
    vector<ConsState>& Unext,
    vector<ConsState>& k1, vector<ConsState>& k2, vector<ConsState>& k3, vector<ConsState>& k4,
    vector<ConsState>& U_temp,
    double dt) 
{
    // Stage 1: 直接將結果寫入預先分配好的 k1
    NSE_Operator(Un, k1);

    // Stage 2
    #pragma omp parallel for
    for (int i = 0; i < Nx * Ny; i++) U_temp[i] = Un[i] + k1[i] * (0.5 * dt);
    NSE_Operator(U_temp, k2);

    // Stage 3
    #pragma omp parallel for
    for (int i = 0; i < Nx * Ny; i++) U_temp[i] = Un[i] + k2[i] * (0.5 * dt);
    NSE_Operator(U_temp, k3);

    // Stage 4
    #pragma omp parallel for
    for (int i = 0; i < Nx * Ny; i++) U_temp[i] = Un[i] + k3[i] * dt;
    NSE_Operator(U_temp, k4);

    // Final Integration
    #pragma omp parallel for
    for (int i = 0; i < Nx * Ny; i++) {
        Unext[i] = Un[i] + (k1[i] + k2[i] * 2.0 + k3[i] * 2.0 + k4[i]) * (dt / 6.0);
    }
}

int main() {
    // 預先分配：這只會執行一次
    vector<ConsState> u0(Nx * Ny), u_next(Nx * Ny);
    vector<ConsState> k1(Nx * Ny), k2(Nx * Ny), k3(Nx * Ny), k4(Nx * Ny), U_temp(Nx * Ny);

    // Evolution loop
    for (int step = 1; step <= 1000; step++) {
        RK4_Step(u0, u_next, k1, k2, k3, k4, U_temp, 0.001);
        u0 = u_next; // 簡單的位移或 swap
        if (step % 20 == 0) {
            cout << "Step: " << step << " | Center density rho = " << u0[idx(Nx/2,Ny/2)].rho << endl;
        }
    }
}