"""
This file includes 2 classes that define the MHE and the Kalman filter-based gradient solver respectively
--------------------------------------------------------------------------------------
Wang Bingheng, at Control and Simulation Lab, ECE Dept. NUS, Singapore
1st version: 31 Aug. 2021
2nd version: 10 May 2022
3rd version: 10 Oct. 2022 after receiving the reviewers' comments
Should you have any question, please feel free to contact the author via:
wangbingheng@u.nus.edu
"""

from casadi import *
from numpy import linalg as LA
import numpy as np
import time as TM

class MHE:
    def __init__(self, horizon, dt_sample):
        self.N = horizon
        self.DT = dt_sample

    def SetStateVariable(self, xa):
        self.state = xa
        self.n_state = xa.numel()

    def SetOutputVariable(self, y):
        assert hasattr(self, 'state'), "Define the state variable first!"
        self.output = y
        self.y_fn   = Function('y',[self.state], [self.output], ['x0'], ['yf'])
        self.n_output = self.output.numel()

    def SetNoiseVariable(self, w):
        self.noise = w
        self.n_noise = w.numel()
    
    def SetQuaternion(self,q):
        self.q = q
        self.n_q = q.numel()

    def SetModelDyn(self, dymh):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'noise'), "Define the noise variable first!"
        # discrete-time dynamic model based on Euler or 4th-order Runge-Kutta method
        self.ModelDyn = self.state + self.DT*dymh
        self.MDyn_fn  = Function('MDyn', [self.state, self.noise], [self.ModelDyn],
                                 ['s','n'], ['MDynf'])

    def SetArrivalCost(self, x_hat):
        assert hasattr(self, 'state'), "Define the state variable first!"
        self.P0        = diag(self.weight_para[0, 0:12])
        # Define an MHE priori
        error_a        = self.state - x_hat # previous mhe estimate at t-N
        self.cost_a    = 1/2 * mtimes(mtimes(transpose(error_a), self.P0), error_a)
        self.cost_a_fn = Function('cost_a', [self.state, self.weight_para], [self.cost_a], ['s','tp'], ['cost_af'])

    def SetCostDyn(self):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'output'), "Define the output variable first!"
        assert hasattr(self, 'noise'), "Define the noise variable first!"
        # Tunable parameters
        self.weight_para = SX.sym('t_para', 1, 26) # dimension: P: 12 + R: 6 + forgetting factor1: 1 + Q: 6 + forgetting factor2: 1 = 25 
        self.n_para      = self.weight_para.numel()
        self.horizon1    = SX.sym('h1') # horizon - 1
        self.horizon2    = self.horizon1 - 1 # horizon - 2
        self.index       = SX.sym('ki')
        self.gamma_r     = self.weight_para[0, 12]
        self.gamma_q     = self.weight_para[0, 19] 
        # r                = horzcat(self.r11[0, 0], self.weight_para[0, 13:18]) 
        r                = self.weight_para[0, 13:19]
        R_t              = diag(r) 
        self.R           = R_t*self.gamma_r**(self.horizon1-self.index)
        self.R_fn        = Function('R_fn', [self.weight_para, self.horizon1, self.index], \
                            [self.R], ['tp','h1', 'ind'], ['R_fnf'])
        Q_t1             = diag(self.weight_para[0, 20:26])
        self.Q           = Q_t1*self.gamma_q**(self.horizon2-self.index)
        # Measurement variable
        self.measurement = SX.sym('ym', self.n_output, 1)

        # Discrete dynamics of the running cost (time-derivative of the running cost) 
        estimtate_error  = self.measurement -self.output
        self.dJ_running  = 1/2*(mtimes(mtimes(estimtate_error.T, self.R), estimtate_error) +
                               mtimes(mtimes(self.noise.T, self.Q), self.noise))
        self.dJ_fn       = Function('dJ_running', [self.state, self.measurement, self.noise, self.weight_para, self.horizon1, self.index],
                              [self.dJ_running], ['s', 'm', 'n', 'tp', 'h1', 'ind'], ['dJrunf'])
        # the terminal cost regarding x_N
        self.dJ_T        = 1/2*mtimes(mtimes(estimtate_error.T, self.R), estimtate_error)
        self.dJ_T_fn     = Function('dJ_T', [self.state, self.measurement, self.weight_para, self.horizon1, self.index], [self.dJ_T],
                                ['s', 'm', 'tp', 'h1', 'ind'], ['dJ_Tf'])

    def MHEsolver(self, Y, x_hat, xmhe_traj, noise_traj, weight_para, time):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'noise'), "Define the noise variable first!"
        assert hasattr(self, 'MDyn_fn'), "Define the model dynamics function first!"
        assert hasattr(self, 'dJ_fn'), "Define the cost dynamics function first!"
        # arrival cost setting
        self.SetArrivalCost(x_hat) # x_hat: MHE estimate at t-N, obtained by the previous MHE
        self.diffKKT_general()
        """
        Formulate MHE as a nonlinear programming problem solved by CasADi nlpsol() function
        """
        # Start with an empty NLP
        w   = [] # optimal trajectory list
        w0  = [] # initial guess of optimal trajectory
        lbw = [] # lower boundary of optimal variables
        ubw = [] # upper boundary of optimal variables
        g   = [] # equality or inequality constraints
        lbg = [] # lower boundary of constraints
        ubg = [] # upper boundary of constraints

        # Initial state for the arrival cost
        Xk  = SX.sym('X0', self.n_state, 1)
        w  += [Xk]
        X_hatmh = []
        l = len(x_hat)
        for i in range(len(x_hat)): # convert an array to a list
            X_hatmh += [x_hat[i,0]]
        w0 += X_hatmh
        lbw+= self.n_state*[-1e20] # value less than or equal to -1e19 stands for no lower bound
        ubw+= self.n_state*[1e20] # value greater than or equal to 1e19 stands for no upper bound
        # Formulate the NLP
        # time_mhe = self.N*self.DT
        if time < self.N: 
            # Full-information estimator
            self.horizon = time + 1 # time starts from 0, so the value of horizon should be larger than time by 1
        else:
            # Moving horizon estimation
            self.horizon = self.N + 1 # note that we start from t-N, so there are N+1 data points

        J = self.cost_a_fn(s=Xk, tp=weight_para)['cost_af']

        for k in range(self.horizon-1):
            # New NLP variables for the process noise
            Nk   = SX.sym('N_' + str(k), self.n_noise, 1)
            w   += [Nk]
            lbw += self.n_noise*[-1e20]
            ubw += self.n_noise*[1e20]
            W_guess = []
            if self.horizon <=3:
                W_guess += self.n_noise*[0]
            else:
                if k<self.horizon-3: # initial guess based on the previous MHE solution
                    for iw in range(self.n_noise):
                        W_guess += [noise_traj[k+1,iw]]
                else:
                    for iw in range(self.n_noise):
                        W_guess += [noise_traj[-1,iw]]
            
            w0  += W_guess 
            # Integrate the cost function till the end of horizon
            J    += self.dJ_fn(s=Xk, m=Y[len(Y)-self.horizon+k], n=Nk, tp=weight_para, h1=self.horizon-1, ind=k)['dJrunf']
            Xnext = self.MDyn_fn(s=Xk,n=Nk)['MDynf']
            # Next state based on the discrete model dynamics and current state
            Xk    = SX.sym('X_' + str(k + 1), self.n_state, 1)
            w    += [Xk]
            lbw  += self.n_state*[-1e20]
            ubw  += self.n_state*[1e20]
            X_guess = []
            if k<self.horizon-3:
                for ix in range(self.n_state):
                    X_guess += [xmhe_traj[k+2, ix]] # due to X_{k+1} which corresponds to X_{k+2} on the previous MHE trajectory
            else:
                for ix in range(self.n_state):
                    X_guess += [xmhe_traj[-1, ix]]
            
            w0 += X_guess
            # Add equality constraint
            g    += [Xk - Xnext] # pay attention to this order! The order should be the same as that defined in the paper!
            lbg  += self.n_state*[0]
            ubg  += self.n_state*[0]

        # Add the final cost
        J += self.dJ_T_fn(s=Xk, m=Y[-1], tp=weight_para, h1=self.horizon-1, ind=self.horizon-1)['dJ_Tf']

        # Create an NLP solver
        opts = {}
        opts['ipopt.tol'] = 1e-8
        opts['ipopt.print_level'] = 0
        opts['print_time'] = 0
        opts['ipopt.warm_start_init_point']='yes'
        opts['ipopt.max_iter']=3e3
        opts['ipopt.acceptable_tol']=1e-7
        opts['ipopt.mu_strategy']='adaptive'
        # opts = {'ipopt.print_level': print_level, 'ipopt.sb': 'yes', 'print_time': print_level}
        prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
        solver = nlpsol('solver', 'ipopt', prob, opts)

        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol['x'].full().flatten() # convert to a row array

        # Take the optimal noise, state, and costate
        sol_traj1 = np.concatenate((w_opt, self.n_noise * [0])) # sol_traj1 = [x0,w0,x1,w1,...,xk,wk,...xn-1,wn-1,xn,wn] note that we added a wn
        sol_traj = np.reshape(sol_traj1, (-1, self.n_state + self.n_noise)) # sol_traj = [[x0,w0],[x1,w1],...[xk,wk],...[xn-1,wn-1],[xn,wn]] 
        state_traj_opt = sol_traj[:, 0:self.n_state] # each xk is a row vector
        noise_traj_opt = np.delete(sol_traj[:, self.n_state:], -1, 0) # delete the last raw as we have added it to make the dimensions of x and w equal
        
        # Compute the co-states using the KKT conditions
        costate_traj_opt = np.zeros((self.horizon, self.n_state))

        for i in range(self.horizon - 1, 0, -1):
            curr_s      = state_traj_opt[i, :]
            curr_n      = noise_traj_opt[i-1,:]
            curr_m      = Y[len(Y) - self.horizon + i]
            lembda_curr = np.reshape(costate_traj_opt[i, :], (self.n_state,1))
            mat_F       = self.F_fn(x0=curr_s, n0=curr_n)['Ff'].full()
            mat_H       = self.H_fn(x0=curr_s)['Hf'].full()
            R_curr      = self.R_fn(tp=weight_para, h1=self.horizon - 1, ind=i)['R_fnf'].full()
            y_curr      = self.y_fn(x0=curr_s)['yf'].full()
            lembda_pre  = np.matmul(np.transpose(mat_F), lembda_curr) + np.matmul(np.matmul(np.transpose(mat_H), R_curr), (curr_m - y_curr))
            costate_traj_opt[(i - 1):i, :] = np.transpose(lembda_pre)
        
        # Alternatively, we can compute the co-states (Lagrange multipliers) from IPOPT itself. These two co-state trajectories are very similar to each other!
        lam_g = sol['lam_g'].full().flatten() # Lagrange multipilers for bounds on g
        costate_traj_ipopt = np.reshape(lam_g, (-1,self.n_state))
        
        # Output
        opt_sol = {"state_traj_opt": state_traj_opt,
                   "noise_traj_opt": noise_traj_opt,
                   "costate_traj_opt": costate_traj_opt,
                   "costate_ipopt": costate_traj_ipopt}
        return opt_sol

    def diffKKT_general(self):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'output'), "Define the output variable first!"
        assert hasattr(self, 'noise'), "Define the noise variable first!"
        assert hasattr(self, 'MDyn_fn'), "Define the model dynamics function first!"
        assert hasattr(self, 'dJ_fn'), "Define the cost dynamics function first!"
        # Define co-state variables
        self.costate      = SX.sym('lambda', self.n_state, 1) # lambda at k
        self.cos_pre      = SX.sym('lampre', self.n_state, 1) # lambda at k-1, in fact, it will not appear in all the 2nd-derivate terms

        # Differentiate the dynamics to get the system Jacobian
        self.F            = jacobian(self.ModelDyn, self.state)
        self.F_fn         = Function('F',[self.state,  self.noise], [self.F], ['x0','n0'], ['Ff']) 
        self.G            = jacobian(self.ModelDyn, self.noise)
        self.G_fn         = Function('G',[self.state,  self.noise], [self.G], ['x0','n0'], ['Gf'])
        self.H            = jacobian(self.output, self.state)
        self.H_fn         = Function('H',[self.state], [self.H], ['x0'], ['Hf'])

        # Definition of Lagrangian
        self.Lbar0        = self.dJ_running - mtimes(transpose(self.costate), self.ModelDyn) # arrival Lagrangian_bar
        self.Lbar_k       = self.dJ_running - mtimes(transpose(self.costate), self.ModelDyn) \
            + mtimes(transpose(self.cos_pre), self.state) # k=t-N+1,...,t-1
        self.L0           = self.cost_a + self.Lbar0 # arrival Lagrangian, k=t-N
        self.LbarT        = self.dJ_T # terminal Lagrangian, k=t

        # First-order derivative of arrival Lagrangian, k=t-N
        self.dL0x         = jacobian(self.L0, self.state) # this is used to calculate ddL0xp
        self.dLbar0x      = jacobian(self.Lbar0, self.state)
        self.dLbar0w      = jacobian(self.Lbar0, self.noise)

        # First-order derivative of path Lagrangian, k=t-N+1,...,t-1
        self.dLbarx       = jacobian(self.Lbar_k, self.state) 
        self.dLbarw       = jacobian(self.Lbar_k, self.noise) 

        # First-order derivative of terminal Lagrangian, k=t
        self.dLbarTx      = jacobian(self.LbarT, self.state)

        # Second-order derivative of arrival Lagrangian, k=t-N
        self.ddL0xp       = jacobian(self.dL0x, self.weight_para)
        self.ddL0xp_fn    = Function('ddL0xp', [self.state, self.costate,  self.noise, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddL0xp], ['x0', 'c0', 'n0', 'm0', 'tp', 'h1', 'ind'], ['ddL0xpf'])
        self.ddLbar0xx    = jacobian(self.dLbar0x, self.state)
        self.ddLbar0xx_fn = Function('ddL0xx', [self.state, self.costate,  self.noise, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddLbar0xx], ['x0', 'c0', 'n0', 'm0', 'tp', 'h1', 'ind'], ['ddL0xxf'])
        self.ddLbar0xw    = jacobian(self.dLbar0x, self.noise)
        self.ddLbar0xw_fn = Function('ddL0xw', [self.state, self.costate,  self.noise, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddLbar0xw], ['x0', 'c0', 'n0', 'm0', 'tp', 'h1', 'ind'], ['ddL0xwf'])
        self.ddLbar0ww    = jacobian(self.dLbar0w, self.noise)
        self.ddLbar0ww_fn = Function('ddL0ww', [self.state, self.costate,  self.noise, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddLbar0ww], ['x0', 'c0', 'n0', 'm0', 'tp', 'h1', 'ind'], ['ddL0wwf'])
        self.ddLbar0wp    = jacobian(self.dLbar0w, self.weight_para)
        self.ddLbar0wp_fn = Function('ddL0wp', [self.state, self.costate,  self.noise, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddLbar0wp], ['x0', 'c0', 'n0', 'm0', 'tp', 'h1', 'ind'], ['ddL0wpf'])
        # note that when k=t-N, ddL0xx = P + ddLbarxx, for all k, ddLxw = ddLbarxw, ddLww = ddLbarww

        # Second-order derivative of path Lagrangian, k=t-N+1,...,t-1
        self.ddLbarxx     = jacobian(self.dLbarx, self.state) 
        self.ddLbarxx_fn  = Function('ddLxx', [self.state, self.costate,  self.noise, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddLbarxx], ['x0', 'c0', 'n0', 'm0', 'tp', 'h1', 'ind'], ['ddLxxf'])
        self.ddLbarxp     = jacobian(self.dLbarx, self.weight_para) 
        self.ddLbarxp_fn  = Function('ddLxp', [self.state, self.costate,  self.noise, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddLbarxp], ['x0', 'c0', 'n0', 'm0', 'tp', 'h1', 'ind'], ['ddLxpf'])
        self.ddLbarxw     = jacobian(self.dLbarx, self.noise) 
        self.ddLbarxw_fn  = Function('ddLxw', [self.state, self.costate,  self.noise, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddLbarxw], ['x0', 'c0', 'n0', 'm0', 'tp', 'h1', 'ind'], ['ddLxwf'])
        self.ddLbarww     = jacobian(self.dLbarw, self.noise) 
        self.ddLbarww_fn  = Function('ddLww', [self.state, self.costate,  self.noise, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddLbarww], ['x0', 'c0', 'n0', 'm0', 'tp', 'h1', 'ind'], ['ddLwwf'])
        self.ddLbarwp     = jacobian(self.dLbarw, self.weight_para) 
        self.ddLbarwp_fn  = Function('ddLwp', [self.state, self.costate,  self.noise, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddLbarwp], ['x0', 'c0', 'n0', 'm0', 'tp', 'h1', 'ind'], ['ddLwpf'])
        
        # Second-order derivative of terminal Lagrangian, k=t
        self.ddLbarTxx    = jacobian(self.dLbarTx, self.state)
        self.ddLbarTxx_fn = Function('ddLTxx', [self.state, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddLbarTxx], ['x0', 'm0', 'tp', 'h1', 'ind'], ['ddLTxxf'])
        self.ddLbarTxp    = jacobian(self.dLbarTx, self.weight_para)
        self.ddLbarTxp_fn = Function('ddLTxp', [self.state, self.measurement, self.weight_para, self.horizon1, self.index], [self.ddLbarTxp], ['x0', 'm0', 'tp', 'h1', 'ind'], ['ddLTxpf'])
   
    def GetAuxSys_general(self, state_traj_opt, costate_traj_opt, noise_traj_opt, weight_para, Y):
        # statement = [hasattr(self, 'A_fn'), hasattr(self, 'D_fn'), hasattr(self, 'E_fn'), hasattr(self, 'F0_fn')]
        horizon = np.size(state_traj_opt, 0)
        self.diffKKT_general()

        # Initialize the coefficient matrices of the auxiliary MHE system:
        matF, matG, matH = [], [], []
        matddLxx, matddLxw, matddLxp, matddLww, matddLwp = [], [], [], [], []

        # Solve the above coefficient matrices
        # Below is applicable only to horizon >1, but it does not matter as we focus on MHE whose horizon is always larger than 1
        for k in range(horizon-1):
            curr_s    = state_traj_opt[k, :] # current state
            curr_cs   = costate_traj_opt[k, :] # current costate, length = horizon, but with the last value being 0
            curr_n    = noise_traj_opt[k,:] # current noise
            curr_m    = Y[len(Y) - horizon + k] # current measurement
            matF     += [self.F_fn(x0=curr_s, n0=curr_n)['Ff'].full()]
            matG     += [self.G_fn(x0=curr_s, n0=curr_n)['Gf'].full()]
            matH     += [self.H_fn(x0=curr_s)['Hf'].full()]
            if k == 0: # note that P is included only in the arrival cost
                matddLxx += [self.ddLbar0xx_fn(x0=curr_s, c0=curr_cs, n0=curr_n, m0=curr_m, tp=weight_para, h1=horizon-1, ind=k)['ddL0xxf'].full()]
                matddLxp += [self.ddL0xp_fn(x0=curr_s, c0=curr_cs, n0=curr_n, m0=curr_m, tp=weight_para, h1=horizon-1, ind=k)['ddL0xpf'].full()] # P,R
                matddLxw += [self.ddLbar0xw_fn(x0=curr_s, c0=curr_cs, n0=curr_n, m0=curr_m, tp=weight_para, h1=horizon-1, ind=k)['ddL0xwf'].full()]
                matddLww += [self.ddLbar0ww_fn(x0=curr_s, c0=curr_cs, n0=curr_n, m0=curr_m, tp=weight_para, h1=horizon-1, ind=k)['ddL0wwf'].full()]
                matddLwp += [self.ddLbar0wp_fn(x0=curr_s, c0=curr_cs, n0=curr_n, m0=curr_m, tp=weight_para, h1=horizon-1, ind=k)['ddL0wpf'].full()] # Q
            else:
                matddLxx += [self.ddLbarxx_fn(x0=curr_s, c0=curr_cs, n0=curr_n, m0=curr_m, tp=weight_para, h1=horizon-1, ind=k)['ddLxxf'].full()]
                matddLxp += [self.ddLbarxp_fn(x0=curr_s, c0=curr_cs, n0=curr_n, m0=curr_m, tp=weight_para, h1=horizon-1, ind=k)['ddLxpf'].full()] # R
                matddLxw += [self.ddLbarxw_fn(x0=curr_s, c0=curr_cs, n0=curr_n, m0=curr_m, tp=weight_para, h1=horizon-1, ind=k)['ddLxwf'].full()]
                matddLww += [self.ddLbarww_fn(x0=curr_s, c0=curr_cs, n0=curr_n, m0=curr_m, tp=weight_para, h1=horizon-1, ind=k)['ddLwwf'].full()]
                matddLwp += [self.ddLbarwp_fn(x0=curr_s, c0=curr_cs, n0=curr_n, m0=curr_m, tp=weight_para, h1=horizon-1, ind=k)['ddLwpf'].full()] # Q
        curr_s    = np.reshape(state_traj_opt[-1, :],(12,1))
        curr_m    = Y[-1]
        output    = np.vstack((np.reshape(curr_s[0:3,0],(3,1)),np.reshape(curr_s[6:9,0],(3,1))))
        error     = curr_m-output
        matddLxx += [self.ddLbarTxx_fn(x0=curr_s, m0=curr_m, tp=weight_para, h1=horizon-1, ind=horizon-1)['ddLTxxf'].full()]  
        matddLxp += [self.ddLbarTxp_fn(x0=curr_s, m0=curr_m, tp=weight_para, h1=horizon-1, ind=horizon-1)['ddLTxpf'].full()] # R, this is incomplete for the full-information estimator when horizon = 1, as the derivative of the arrival cost is missing.

        auxSys = {"matF": matF,
                  "matG": matG,
                  "matH": matH,
                  "matddLxx": matddLxx,
                  "matddLxp": matddLxp,
                  "matddLxw": matddLxw,
                  "matddLww": matddLww,
                  "matddLwp": matddLwp
                 }
        return auxSys
    
    def diffKKT_second_general(self,X_hat):
        self.diffKKT_general()
        self.X         = SX.sym('Xg',self.n_state,self.n_para)
        self.Xnext     = SX.sym('Xnext',self.n_state,self.n_para)
        self.W         = SX.sym('Wg',self.n_noise,self.n_para)
        self.Lambda    = SX.sym('Lam',self.n_state,self.n_para)
        self.preLambda = SX.sym('pLam',self.n_state,self.n_para) # this will not be used in the derivation, but included here for completeness.

        # Initial fbar_0 at k=t-N
        self.fbar_0 = (self.P0 + self.ddLbar0xx)@self.X - self.P0@X_hat + self.ddLbar0xw@self.W - self.F.T@self.Lambda + self.ddL0xp
        
        # Remaining fbar for k=t-N+1, ..., t-1
        self.fbar_k = self.ddLbarxx@self.X + self.ddLbarxw@self.W - self.F.T@self.Lambda + self.preLambda + self.ddLbarxp

        # Termial fbar at k=t
        self.fbar_t = self.ddLbarTxx@self.X + self.preLambda + self.ddLbarTxp

        # gbar for k=t-N, ..., t-1
        self.gbar_k = self.ddLbarxw.T@self.X + self.ddLbarww@self.W - self.G.T@self.Lambda + self.ddLbarwp 

        # hbar for k=t-N, ..., t-1
        self.hbar_k = self.Xnext - self.F@self.X - self.G@self.W 
        # In the above definitions, we should continue to use those 2nd-order coefficient matrices defined in the function 'diffKKT_general'. 
        # Otherwise, the following partial derivatives will be incorrect as the dependence of those coefficient matrrices on the system states and costates is missing.

        # Partial derivatives of fbar_0 at k=t-N
        dfbar0dx    = jacobian(self.fbar_0,self.state)
        dfbar0dc    = jacobian(self.fbar_0,self.costate)
        dfbar0dw    = jacobian(self.fbar_0,self.noise)
        dfbar0dp    = jacobian(self.fbar_0,self.weight_para)
        self.Dfbar0dp    = dfbar0dx@self.X + dfbar0dc@self.Lambda + dfbar0dw@self.W + dfbar0dp
        self.Dfbar0dp_fn = Function('Dfbar0dp',[self.X,self.Lambda,self.W,self.state,self.costate,self.noise,self.measurement,self.weight_para,self.horizon1,self.index],[self.Dfbar0dp],['X0','C0','W0','x0','c0','n0','m0','tp','h1','ind'],['Dfbar0dpf'])


        # Partial derivatives of fbar for k=t-N+1, ..., t-1
        dfbardx     = jacobian(self.fbar_k,self.state)
        dfbardc     = jacobian(self.fbar_k,self.costate)
        dfbardw     = jacobian(self.fbar_k,self.noise)
        dfbardp     = jacobian(self.fbar_k,self.weight_para)
        self.Dfbarkdp    = dfbardx@self.X + dfbardc@self.Lambda + dfbardw@self.W + dfbardp
        self.Dfbarkdp_fn = Function('Dfbarkdp',[self.X,self.Lambda,self.W,self.state,self.costate,self.noise,self.measurement,self.weight_para,self.horizon1,self.index],[self.Dfbarkdp],['X0','C0','W0','x0','c0','n0','m0','tp','h1','ind'],['Dfbarkdpf'])

        # Partial derivative of fbar at k=t
        dfbarTdx    = jacobian(self.fbar_t,self.state)
        dfbarTdp    = jacobian(self.fbar_t,self.weight_para)
        self.DfbarTdp    = dfbarTdx@self.X + dfbarTdp
        self.DfbarTdp_fn = Function('DfbarTdp',[self.X,self.state,self.measurement,self.weight_para,self.horizon1,self.index],[self.DfbarTdp],['X0','x0','m0','tp','h1','ind'],['DfbarTdpf'])
        
        # Partial derivatives of gbar for k=t-N, ..., t-1
        dgbardx     = jacobian(self.gbar_k,self.state)
        dgbardc     = jacobian(self.gbar_k,self.costate)
        dgbardw     = jacobian(self.gbar_k,self.noise)
        dgbardp     = jacobian(self.gbar_k,self.weight_para)
        self.Dgbarkdp    = dgbardx@self.X + dgbardc@self.Lambda + dgbardw@self.W + dgbardp
        self.Dgbarkdp_fn = Function('Dgbarkdp',[self.X,self.Lambda,self.W,self.state,self.costate,self.noise,self.measurement,self.weight_para,self.horizon1,self.index],[self.Dgbarkdp],['X0','C0','W0','x0','c0','n0','m0','tp','h1','ind'],['Dgbarkdpf'])

        # Partial derivatives of hbar for k=t-N, ..., t-1
        dhbardx     = jacobian(self.hbar_k,self.state)
        dhbardw     = jacobian(self.hbar_k,self.noise)
        self.Dhbarkdp    = dhbardx@self.X + dhbardw@self.W 
        self.Dhbarkdp_fn = Function('Dhbarkdp',[self.X,self.W,self.state,self.noise],[self.Dhbarkdp],['X0','W0','x0','n0'],['Dhbarkdpf'])


    def GetAuxSys_second_general(self,X_hat,state_traj_opt, costate_traj_opt, noise_traj_opt, weight_para, Y, X_traj_opt, Lambda_traj_opt, W_traj_opt):
        self.diffKKT_second_general(X_hat)
        horizon = np.size(state_traj_opt, 0)
        # Initialize the coefficient matrices of the 2nd-order auxiliary MHE system:
        matDfbar, matDgbar, matDhbar = [],[],[]
    
        # Solve the above coefficient matrices
        for k in range(horizon-1):
            curr_s    = state_traj_opt[k,:]
            curr_cs   = costate_traj_opt[k, :] # current costate, length = horizon, but with the last value being 0
            curr_n    = noise_traj_opt[k,:] # current noise
            curr_m    = Y[len(Y) - horizon + k] # current measurement
            curr_X    = X_traj_opt[k]
            curr_CS   = Lambda_traj_opt[k]
            curr_W    = W_traj_opt[k]
            if k==0:
                matDfbar += [self.Dfbar0dp_fn(X0=curr_X,C0=curr_CS,W0=curr_W,x0=curr_s,c0=curr_cs,n0=curr_n,m0=curr_m,tp=weight_para,h1=horizon-1,ind=k)['Dfbar0dpf'].full()]
            else:
                matDfbar += [self.Dfbarkdp_fn(X0=curr_X,C0=curr_CS,W0=curr_W,x0=curr_s,c0=curr_cs,n0=curr_n,m0=curr_m,tp=weight_para,h1=horizon-1,ind=k)['Dfbarkdpf'].full()]
            matDgbar += [self.Dgbarkdp_fn(X0=curr_X,C0=curr_CS,W0=curr_W,x0=curr_s,c0=curr_cs,n0=curr_n,m0=curr_m,tp=weight_para,h1=horizon-1,ind=k)['Dgbarkdpf'].full()]
            matDhbar += [self.Dhbarkdp_fn(X0=curr_X,W0=curr_W,x0=curr_s,n0=curr_n)['Dhbarkdpf'].full()]
        curr_X    = X_traj_opt[-1]
        curr_s    = state_traj_opt[-1,:]
        curr_m    = Y[-1]
        matDfbar += [self.DfbarTdp_fn(X0=curr_X,x0=curr_s,m0=curr_m,tp=weight_para,h1=horizon-1,ind=horizon-1)['DfbarTdpf'].full()]
        
        auxSys_2nd = {"matDfbar": matDfbar,
                      "matDgbar": matDgbar,
                      "matDhbar": matDhbar
                      }
        
        return auxSys_2nd

        

    def Observability(self, matF, matH):
        rank = np.zeros(len(matF))
        n = len(matF)
        for i in range(n):
            ok0 = np.matmul(matH[i],LA.matrix_power(matF[i],0))
            O   = ok0.flatten()
            for k in range(1,self.n_state):
                ok= np.matmul(matH[i],LA.matrix_power(matF[i],k))
                o = ok.flatten()
                O = np.concatenate((O,o))
            O = np.reshape(O,(-1,self.n_state))
            rank_i = LA.matrix_rank(O)
            rank[i] = rank_i
        return rank
    
    def Hessian_Lagrange(self, matddLxx, matddLxw, matddLww):
        horizon   = len(matddLxx)
        min_eig_k = np.zeros(horizon)
        min4_eig_k = np.zeros(horizon)
        for k in range(horizon-1):
            H_k = np.vstack((
                np.hstack((matddLxx[k],matddLxw[k])),
                np.hstack((matddLxw[k].T,matddLww[k]))
            ))
            # H_k = np.vstack((
            #     np.hstack((matddLxx[k],np.zeros((19,6)))),
            #     np.hstack((np.zeros((19,6)).T,matddLww[k]))
            # ))
            w, v = LA.eig(H_k)
            min_eig_k[k] = np.min(np.real(w))
            min4_eig_k[k] = np.sort(np.real(w))[5]
            inv_H_k = LA.inv(matddLww[k]) # H_k is not guaranteed to be positive-definite and nonsingular
        w, v = LA.eig(matddLxx[-1])
        # min_eig_k[-1] = np.min(np.real(w))
        min_eig = np.min(min_eig_k)
        # min4_eig_k[-1] = np.sort(np.real(w))[2]
        min4_eig = np.max(min4_eig_k)
        return min_eig, min4_eig

class Trust_region:
    def __init__(self, nn_para):
        self.n_para  = nn_para.numel()
        self.wchange = SX.sym('mu_t',1,self.n_para)
        # self.grad    = SX.sym('grad',1,self.n_para)
        # self.hessian = SX.sym('hessian',self.n_para,self.n_para)
        # self.loss    = SX.sym('loss')
    
    # def SetCost_TRS(self):
    #     self.ltrs = self.loss + self.grad@self.wchange.T + 1/2*self.wchange@self.hessian@self.wchange.T
    #     self.ltrs_fn = Function('lTRS',[self.wchange, self.loss, self.grad, self.hessian],[self.ltrs],['mu0','l0','grad0','hess0'],['lTRSf'])
    
    def TRS_solver(self, Grad_dldn, Hess_dldn, radius):
        """
        Formulate TRS as a nonlinear programming problem solved by CasADi nlpsol() function
        """
        # Start with an empty NLP
        w   = [] # optimal trajectory list
        w0  = [] # initial guess of optimal trajectory
        lbw = [] # lower boundary of optimal variables
        ubw = [] # upper boundary of optimal variables
        g   = [] # equality or inequality constraints
        lbg = [] # lower boundary of constraints
        ubg = [] # upper boundary of constraints

        # Decision variable: the change of the tuning parameters
        wchange = SX.sym('wc',1,self.n_para)
        w   += [wchange]
        w0  += self.n_para*[0]
        # Lower bound of the decision variable change
        lbw += self.n_para*[-1e20]
        # Upper bound of the decision variable change
        ubw += self.n_para*[1e20]
        # Formulate the NLP
        ltrs = Grad_dldn@wchange.T + 1/2*wchange@Hess_dldn@wchange.T
        ltrs_fn = Function('lTRS',[wchange],[ltrs],['mu0'],['lTRSf'])
        J    = ltrs_fn(mu0=wchange)['lTRSf']
        # Inequality constraint
        g   += [wchange@wchange.T] #norm_2(wchange)
        lbg += [0]
        ubg += [radius**2]
      
        # Create an NLP solver
        opts = {}
        # opts['ipopt.tol'] = 1e-8
        opts['ipopt.print_level'] = 0
        opts['print_time'] = 0
        opts['ipopt.warm_start_init_point']='yes'
        opts['ipopt.max_iter']=1e2
        # opts['ipopt.acceptable_tol']=1e-7
        opts['ipopt.ma57_automatic_scaling']='yes'
        opts['ipopt.mu_strategy']='adaptive'
        # opts = {'ipopt.print_level': print_level, 'ipopt.sb': 'yes', 'print_time': print_level}
        prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
        solver = nlpsol('solver', 'ipopt', prob, opts)

        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        wc_opt = sol['x'].full().flatten() # convert to a row array
        TRS_opt= sol['f'].full()
        return wc_opt, TRS_opt
    
    def TRS_solver_SOCP(self, Grad_dldn, Hess_dldn, radius):
        """
        Formulate TRS as a nonlinear programming problem solved by CasADi nlpsol() function
        """
        # Start with an empty NLP
        w   = [] # optimal trajectory list
        w0  = [] # initial guess of optimal trajectory
        lbw = [] # lower boundary of optimal variables
        ubw = [] # upper boundary of optimal variables
        g   = [] # equality or inequality constraints
        lbg = [] # lower boundary of constraints
        ubg = [] # upper boundary of constraints

        # Decision variable: the change of the tuning parameters
        wchange = SX.sym('wc',1,self.n_para)
        tx      = SX.sym('tx',1,1)
        w   += [wchange]
        w   += [tx]
        w0  += self.n_para*[0]
        w0  += [0]
        # Lower bound of the decision variable change
        lbw += self.n_para*[-1e20]
        lbw += [0]
        # Upper bound of the decision variable change
        ubw += self.n_para*[1e20]
        ubw += [1e20]
        #----Make the Hessian symmatric----#
        Hess_dldn = (Hess_dldn + Hess_dldn.T)/2
        # Formulate the NLP
        ltrs_socp    = Grad_dldn@wchange.T + tx
        ltrs_socp_fn = Function('ltrs_socp',[wchange,tx],[ltrs_socp],['mu0','t0'],['ltrsf'])
        J    = ltrs_socp_fn(mu0=wchange,t0=tx)['ltrsf']
        # Inequality constraint
        g   += [wchange@wchange.T] #norm_2(wchange)
        lbg += [0]
        ubg += [radius**2]
        g   += [1/2*wchange@Hess_dldn@wchange.T-tx]
        lbg += [-1e20]
        ubg += [0]

        # Create an NLP solver
        opts = {}
        # opts['ipopt.tol'] = 1e-8
        opts['ipopt.print_level'] = 0
        opts['print_time'] = 0
        opts['ipopt.warm_start_init_point']='yes'
        opts['ipopt.max_iter']=1e2
        # opts['ipopt.acceptable_tol']=1e-7
        opts['ipopt.ma57_automatic_scaling']='yes'
        opts['ipopt.mu_strategy']='adaptive'
        # opts = {'ipopt.print_level': print_level, 'ipopt.sb': 'yes', 'print_time': print_level}
        prob = {'f': J, 'x': horzcat(*w), 'g': horzcat(*g)}
        solver = nlpsol('solver', 'ipopt', prob, opts)

        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        sol_traj = sol['x'].full().flatten() # convert to a row array
        sol_traj = np.reshape(sol_traj,(1,self.n_para+1))
        wc_opt   = sol_traj[0,0:self.n_para]
        TRS_opt  = sol['f'].full()
        return wc_opt, TRS_opt
    
    def TRS_solver_Eigen(self, loss, Grad_dldn, Hess_dldn, radius):
        """
        Formulate TRS as a nonlinear programming problem solved by CasADi nlpsol() function
        """
        # Start with an empty NLP
        w   = [] # optimal trajectory list
        w0  = [] # initial guess of optimal trajectory
        lbw = [] # lower boundary of optimal variables
        ubw = [] # upper boundary of optimal variables
        g   = [] # equality or inequality constraints
        lbg = [] # lower boundary of constraints
        ubg = [] # upper boundary of constraints

        # Decision variable: the change of the tuning parameters
        wchange = SX.sym('wc',1,self.n_para)
        w   += [wchange]
        w0  += self.n_para*[0]
        # Lower bound of the decision variable change
        lbw += self.n_para*[-1e20]
        # Upper bound of the decision variable change
        ubw += self.n_para*[1e20]
        # Eigenvalue decomposition of Hess_dldn
        #----Make the Hessian symmatric----#
        Hess_dldn = (Hess_dldn + Hess_dldn.T)/2
        Eigenvalues, Eigenvectors = LA.eig(Hess_dldn) # take their real parts if complex eigenvalues are encountered for the inversed Kronecker product
        Lambda = np.diag(Eigenvalues)

        # Formulate the NLP
        ltrs =loss + Grad_dldn@Eigenvectors@wchange.T + 1/2*wchange@Lambda@wchange.T
        ltrs_fn = Function('lTRS',[wchange],[ltrs],['mu0'],['lTRSf'])
        J    = ltrs_fn(mu0=wchange)['lTRSf']
        # Inequality constraint
        g   += [wchange@wchange.T] #norm_2(wchange)
        lbg += [0]
        ubg += [radius**2]
      
        # Create an NLP solver
        opts = {}
        # opts['ipopt.tol'] = 1e-8
        opts['ipopt.print_level'] = 0
        opts['print_time'] = 0
        opts['ipopt.warm_start_init_point']='yes'
        opts['ipopt.max_iter']=1e2
        # opts['ipopt.acceptable_tol']=1e-7
        opts['ipopt.ma57_automatic_scaling']='yes'
        opts['ipopt.mu_strategy']='adaptive'
        # opts = {'ipopt.print_level': print_level, 'ipopt.sb': 'yes', 'print_time': print_level}
        prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
        solver = nlpsol('solver', 'ipopt', prob, opts)

        # Solve the NLP
        sol     = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        x_opt   = sol['x'].full().flatten() # convert to a row array
        TRS_opt = sol['f'].full()
        wc_opt  = x_opt@Eigenvectors.T

        return wc_opt, TRS_opt

    def Get_ratio(self,loss,loss_updated,TRS_opt):
        ratio = (loss - loss_updated)/(loss - TRS_opt)
        return ratio
    
    def TRS_radius(self,wc_opt,ratio,radius,t2,t3,chi1,chi2,upperb):
        if ratio<t2:
            radius_new = chi1*radius
        elif ratio>t3 and LA.norm(wc_opt) == radius:
            radius_new = np.minimum(chi2*radius,upperb)
        else:
            radius_new = radius
        return radius_new

class Net:
    def __init__(self, D_in, D_h, D_out):
        self.nn_para = SX.sym('nnp',1,(D_h*2+D_out+D_in*D_h+D_h**2+D_h*D_out))
        self.D_in    = D_in
        self.D_h     = D_h
        self.D_out   = D_out
        self.b_1     = self.nn_para[0,0:D_h].T
        self.b_2     = self.nn_para[0,D_h:(2*D_h)].T
        self.b_o     = self.nn_para[0,(2*D_h):(D_h*2+D_out)].T
        self.A_1     = SX.sym('A1',D_h,D_in)
        for k in range(D_h):
            ak = self.nn_para[0,(D_h*2+D_out+k*D_in):(D_h*2+D_out+(k+1)*D_in)]
            self.A_1[k,:] = ak
        self.A_1_fn  = Function('A1',[self.nn_para],[self.A_1],['nnp0'],['A1f'])
        self.A_2     = SX.sym('A2',D_h,D_h)
        for k in range(D_h):
            ak = self.nn_para[0,(D_h*2+D_out+D_in*D_h+k*D_h):(D_h*2+D_out+D_in*D_h+(k+1)*D_h)]
            self.A_2[k,:] = ak
        self.A_2_fn  = Function('A2',[self.nn_para],[self.A_2],['nnp0'],['A2f'])
        self.A_o     = SX.sym('Ao',D_out,D_h)
        for k in range(D_out):
            ak = self.nn_para[0,(D_h*2+D_out+D_h**2+D_in*D_h+k*D_h):(D_h*2+D_out+D_h**2+D_in*D_h+(k+1)*D_h)]
            self.A_o[k,:] = ak
        self.A_o_fn  = Function('Ao',[self.nn_para],[self.A_o],['nnp0'],['Aof'])

    def Forward(self,input,nn_para):
        # Slope in the Leaky_ReLU activation
        alpha = 1e-2
        # hidden layer 1
        A_1_nv = self.A_1_fn(nnp0=nn_para)['A1f'].full()
        U,S,Vh = LA.svd(A_1_nv)
        sigma1 = np.max(S)
        z1_sym = self.A_1@input/sigma1 + self.b_1
        z1_fun = Function('z1',[self.nn_para],[z1_sym],['nnp0'],['z1f'])
        z1     = z1_fun(nnp0=nn_para)['z1f'].full()
        z1     = np.reshape(z1,(self.D_h,1))
        z2_sym = SX.sym('z2',self.D_h,1)
        z2     = np.zeros((self.D_h,1))
        for k in range(self.D_h):
            # Leaky_ReLU activation
            if z1[k,0]>=0:
                z2_sym[k,0] = z1_sym[k,0]
                z2[k,0]     = z1[k,0]
            else:
                z2_sym[k,0] = alpha*z1_sym[k,0]
                z2[k,0]     = alpha*z1[k,0]
        # hidden layer 2
        A_2_nv = self.A_2_fn(nnp0=nn_para)['A2f'].full()
        U,S,Vh = LA.svd(A_2_nv)
        sigma2 = np.max(S)
        z3_sym = self.A_2@z2_sym/sigma2 + self.b_2
        z3_fun = Function('z3',[self.nn_para],[z3_sym],['nnp0'],['z3f'])
        z3     = z3_fun(nnp0=nn_para)['z3f'].full()
        z3     = np.reshape(z3,(self.D_h,1))
        z4_sym = SX.sym('z4',self.D_h,1)
        z4     = np.zeros((self.D_h,1))
        for k in range(self.D_h):
            # Leaky_ReLU activation
            if z3[k,0]>=0:
                z4_sym[k,0] = z3_sym[k,0]
                z4[k,0]     = z3[k,0]
            else:
                z4_sym[k,0] = alpha*z3_sym[k,0]
                z4[k,0]     = alpha*z3[k,0]
        # output layer
        A_o_nv = self.A_o_fn(nnp0=nn_para)['Aof'].full()
        U,S,Vh = LA.svd(A_o_nv)
        sigmao = np.max(S)
        z5_sym = self.A_o@z4_sym/sigmao + self.b_o
        z5_fun = Function('z5',[self.nn_para],[z5_sym],['nnp0'],['z5f'])
        z5     = z5_fun(nnp0=nn_para)['z5f'].full()
        return z5, z5_sym
    
    def Backward(self,z5_sym,nn_para):
        start_time = TM.time()
        nn_grad     = jacobian(z5_sym,self.nn_para)
        gradtime = (TM.time() - start_time)
        # print("back_gradtime=--- %s s ---" % format(gradtime,'.2f'))
        nn_grad_fn  = Function('nn_grad',[self.nn_para],[nn_grad],['nnp0'],['nngf'])
        neural_grad = nn_grad_fn(nnp0=nn_para)['nngf'].full()
        start_time = TM.time()
        nn_hess     = jacobian(nn_grad,self.nn_para)
        hesstime = (TM.time() - start_time)
        # print("back_hesstime=--- %s s ---" % format(hesstime,'.2f'))
        nn_hess_fn  = Function('nn_hess',[self.nn_para],[nn_hess],['nnp0'],['nnhf'])
        start_time = TM.time()
        neural_hess = nn_hess_fn(nnp0=nn_para)['nnhf'].full()
        hess_comp_time = (TM.time() - start_time)
        # print("back_hess_comp_time=--- %s s ---" % format(hess_comp_time,'.2f'))

        return neural_grad, neural_hess



"""
The KF_gradient_solver class solves for the explicit solutions of the gradients of optimal trajectories
w.r.t the tunable parameters 
"""
class KF_gradient_solver:
    def __init__(self, xa, para, w, nn_para):
        self.n_xmhe = xa.numel()
        self.n_para = para.numel()
        self.n_wmhe = w.numel()
        self.n_neur = nn_para.numel()
        self.x_t    = SX.sym('x_t',6,1)
        self.xa     = xa
        self.dismhe = vertcat(self.xa[3:6,0], self.xa[9:12,0]) 
        self.est_e  = self.dismhe - self.x_t
        w_f, w_t    = 1, 1
        weight      = np.array([w_f, w_f, 2*w_f, w_t, w_t, w_t])
        self.loss   = mtimes(mtimes(transpose(self.est_e), np.diag(weight)), self.est_e)
        self.Kloss  = 1

    def GradientSolver_general(self, Xhat, auxSys, weight_para): 
        matF, matG = auxSys['matF'], auxSys['matG']
        matddLxx, matddLxp = auxSys['matddLxx'], auxSys['matddLxp']
        matddLxw, matddLww, matddLwp = auxSys['matddLxw'], auxSys['matddLww'], auxSys['matddLwp']
        self.horizon = len(matddLxx)
        pdiag   = np.reshape(weight_para[0,0:12],(1,12))
        P0      = np.diag(pdiag[0])


        """-------------------------Forward Kalman filter-----------------------------"""
        # Initialize the state and covariance matrix
        X_KF    = self.horizon*[np.zeros((self.n_xmhe, self.n_para))]
        C       = self.horizon*[np.zeros((self.n_xmhe, self.n_xmhe))]
        S       = self.horizon*[np.zeros((self.n_xmhe, self.n_xmhe))]
        T       = self.horizon*[np.zeros((self.n_xmhe, self.n_xmhe))]
        F_bar   = (self.horizon-1)*[np.zeros((self.n_xmhe, self.n_xmhe))]
        if self.horizon == 1: 
            S_k = -matddLxx[0]
            T_k = -matddLxp[0]
        else:
            S_k = np.matmul(np.matmul(matddLxw[0], LA.inv(matddLww[0])), np.transpose(matddLxw[0]))-matddLxx[0]
            T_k = np.matmul(np.matmul(matddLxw[0], LA.inv(matddLww[0])), matddLwp[0])-matddLxp[0]
        S[0]    = S_k
        T[0]    = T_k
        P_k     = LA.inv(P0)
        C_k     = np.matmul(LA.inv(np.identity(self.n_xmhe)-np.matmul(P_k, S_k)), P_k)
        C[0]    = C_k
        X_KFk   = np.matmul((np.identity(self.n_xmhe)+np.matmul(C_k, S_k)), Xhat) + np.matmul(C_k,T_k)
        X_KF[0] = X_KFk

        F_bartime, X_kk1time, P_k1time, S_k1time=0,0,0,0
        C_k1time, T_k1time, X_KFk1time, LAMBDA_time, X_optk_time = 0,0,0,0,0
        
        for k in range(self.horizon-1):
            F_bark    = matF[k]-np.matmul(np.matmul(matG[k], LA.inv(matddLww[k])), np.transpose(matddLxw[k]))
            F_bar[k]  = F_bark
            # state predictor
            X_kk1     = np.matmul(F_bar[k], X_KF[k]) - np.matmul(np.matmul(matG[k], LA.inv(matddLww[k])), matddLwp[k]) # X_kk1: X^hat_{k+1|k}
            # error covariance
            P_k1       = np.matmul(np.matmul(F_bar[k], C[k]), np.transpose(F_bar[k])) + np.matmul(np.matmul(matG[k], LA.inv(matddLww[k])), np.transpose(matG[k])) # P_k1: P_{k+1}
            
            # Kalman gain
            if k < self.horizon-2:
                S_k1   = np.matmul(np.matmul(matddLxw[k+1], LA.inv(matddLww[k+1])), np.transpose(matddLxw[k+1]))-matddLxx[k+1] # S_k1: S_{k+1}
                
            else:
                S_k1   = -matddLxx[k+1] # matddLxw_{t-N} does not exist
            S[k+1]    = S_k1
            C_k1      = np.matmul(LA.inv(np.identity(self.n_xmhe)-np.matmul(P_k1, S[k+1])), P_k1)
            
            C[k+1]    = C_k1
            
            # state corrector
            if k < self.horizon-2:
                T_k1   = np.matmul(np.matmul(matddLxw[k+1], LA.inv(matddLww[k+1])), matddLwp[k+1])-matddLxp[k+1] # T_k1: T_{k+1}
                
            else:
                T_k1   = -matddLxp[k+1]
            T[k+1]    = T_k1
            X_KFk1    = np.matmul((np.identity(self.n_xmhe)+np.matmul(C[k+1], S[k+1])), X_kk1) + np.matmul(C[k+1], T[k+1])
            
            X_KF[k+1] = X_KFk1
        # if self.horizon>10:
        #     print("F_bartime=--- %s ms ---" % format(F_bartime/(self.horizon-1),'.2f'))
        #     print("X_kk1time=--- %s ms ---" % format(X_kk1time/(self.horizon-1),'.2f'))
        #     print("P_k1time=--- %s ms ---" % format(P_k1time/(self.horizon-1),'.2f'))
        #     print("S_k1time=--- %s ms ---" % format(S_k1time/(self.horizon-1),'.2f'))
        #     print("C_k1time=--- %s ms ---" % format(C_k1time/(self.horizon-1),'.2f'))
        #     print("T_k1time=--- %s ms ---" % format(T_k1time/(self.horizon-1),'.2f'))
        #     print("X_KFk1time=--- %s ms ---" % format(X_KFk1time/(self.horizon-1),'.2f'))

     
        """-------------------------Backward costate gradient--------------------------"""
        LAMBDA      = self.horizon*[np.zeros((self.n_xmhe, self.n_para))]
        Lambda_last = np.zeros((self.n_xmhe, self.n_para))
        LAMBDA[-1]  = Lambda_last

        for k in range((self.horizon-1), 0, -1):
            if k == self.horizon-1: # the length of F_bar is (horizon - 1)
                LAMBDA_pre = np.matmul(S[k], X_KF[k]) + T[k]
            else:
                LAMBDA_pre = np.matmul((np.identity(self.n_xmhe)+np.matmul(S[k], C[k])), np.matmul(np.transpose(F_bar[k]), LAMBDA[k])) + np.matmul(S[k], X_KF[k]) + T[k]
                
            LAMBDA[k-1] = LAMBDA_pre
        # if self.horizon>10:
        #     print("LAMBDA_time=--- %s ms ---" % format(LAMBDA_time/(self.horizon-1),'.2f'))
        """-------------------------Forward state gradient-----------------------------"""
        X_opt = self.horizon*[np.zeros((self.n_xmhe, self.n_para))]
        for k in range(self.horizon):
            if k < self.horizon-1:
                X_optk  = X_KF[k] + np.matmul(np.matmul(C[k], np.transpose(F_bar[k])), LAMBDA[k])
                
            else:
                X_optk  = X_KF[k]
            X_opt[k] = X_optk
        # if self.horizon>10:
        #     print("X_optk_time=--- %s ms ---" % format(X_optk_time/(self.horizon-1),'.2f'))
        W_opt = (self.horizon-1)*[np.zeros((self.n_wmhe, self.n_para))]

        for k in range(self.horizon-1):
            W_opt[k] = LA.inv(matddLww[k]) @ (np.transpose(matG[k]) @ LAMBDA[k] - matddLwp[k] - np.transpose(matddLxw[k]) @ X_opt[k])

        gra_opt = {"state_gra_traj": X_opt,
                   "costate_gra_traj": LAMBDA,
                   "noise_gra_traj": W_opt
                   }  
        return gra_opt

    def GradientSolver_2nd_general(self,HXhat,auxSys,auxSys_2nd,weight_para):
        matF, matG = auxSys['matF'], auxSys['matG']
        matddLxx, matddLxp = auxSys['matddLxx'], auxSys['matddLxp']
        matddLxw, matddLww, matddLwp = auxSys['matddLxw'], auxSys['matddLww'], auxSys['matddLwp']
        matDfbar, matDgbar, matDhbar = auxSys_2nd['matDfbar'], auxSys_2nd['matDgbar'], auxSys_2nd['matDhbar']
        horizon = len(matddLxx)
        pdiag   = np.reshape(weight_para[0,0:12],(1,12))
        P0      = np.diag(pdiag[0])
        I_n     = np.identity(self.n_para)
        I_nm    = np.identity(self.n_para*self.n_xmhe)
        """---------------------Forward Kalman filter for 2nd-order derivatives--------------------"""
        # Initialization
        HpX_KF  = horizon*[np.zeros((self.n_para*self.n_xmhe,self.n_para))]
        C_H     = horizon*[np.zeros((self.n_para*self.n_xmhe,self.n_para*self.n_xmhe))]
        S_H     = horizon*[np.zeros((self.n_para*self.n_xmhe,self.n_para*self.n_xmhe))]
        T_H     = horizon*[np.zeros((self.n_para*self.n_xmhe,self.n_para))]
        Fbar_H  = (horizon-1)*[np.zeros((self.n_para*self.n_xmhe,self.n_para*self.n_xmhe))]
        if horizon == 1: # full-information estimator
            SH_k = -np.kron(I_n,matddLxx[0])
            TH_k = -matDfbar[0]
        else:
            SH_k = np.kron(I_n,matddLxw[0])@np.kron(I_n,LA.inv(matddLww[0]))@np.kron(I_n,np.transpose(matddLxw[0])) - np.kron(I_n,matddLxx[0])
            TH_k = np.kron(I_n,matddLxw[0])@np.kron(I_n,LA.inv(matddLww[0]))@matDgbar[0] - matDfbar[0]
        S_H[0]   = SH_k
        T_H[0]   = TH_k
        PH_k     = np.kron(I_n,LA.inv(P0))
        CH_k     = LA.inv(I_nm - PH_k@SH_k)@PH_k
        C_H[0]   = CH_k
        HpX_KFk  = (I_nm + CH_k@SH_k)@HXhat + CH_k@TH_k
        HpX_KF[0]= HpX_KFk

        Fbar_Hktime, HpX_kk1time, PH_k1time, SH_k1time= 0,0,0,0
        CH_k1time, TH_k1time, HpX_KFk1time, HpLAMBDA_time, HpX_optk_time = 0,0,0,0,0
        for k in range(horizon-1):
            Fbar_Hk   = np.kron(I_n,matF[k]) - np.kron(I_n,matG[k])@np.kron(I_n,LA.inv(matddLww[k]))@np.kron(I_n,np.transpose(matddLxw[k]))
            
            Fbar_H[k] = Fbar_Hk
            # state predictor
            HpX_kk1   = Fbar_H[k]@HpX_KF[k] - np.kron(I_n,matG[k])@np.kron(I_n,LA.inv(matddLww[k]))@matDgbar[k] - matDhbar[k] # HpX_kk1: HpX_{k+1|k}

            # error covariance
            PH_k1     = Fbar_H[k]@C_H[k]@np.transpose(Fbar_H[k]) + np.kron(I_n,matG[k])@np.kron(I_n,LA.inv(matddLww[k]))@np.kron(I_n,np.transpose(matG[k])) # PH_k1: PH_{k+1}
            
            # corrector of the error covariance
            if k  < horizon-2:
                SH_k1  = np.kron(I_n,matddLxw[k+1])@np.kron(I_n,LA.inv(matddLww[k+1]))@np.kron(I_n,np.transpose(matddLxw[k+1])) - np.kron(I_n,matddLxx[k+1])
                
            else:
                SH_k1  = - np.kron(I_n,matddLxx[k+1])# matddLxw_{t-N} does not exist
            S_H[k+1]  = SH_k1
            CH_k1     = LA.inv(I_nm - PH_k1@S_H[k+1])@PH_k1
            
            C_H[k+1]  = CH_k1
            # state corrector
            if k < horizon -2:
                TH_k1  = np.kron(I_n,matddLxw[k+1])@np.kron(I_n,LA.inv(matddLww[k+1]))@matDgbar[k+1] - matDfbar[k+1] #TH_k1: TH_{k+1}
                
            else:
                TH_k1  = - matDfbar[k+1]
            T_H[k+1]  = TH_k1
            HpX_KFk1  = (I_nm + C_H[k+1]@S_H[k+1])@HpX_kk1 + C_H[k+1]@T_H[k+1]
            
            HpX_KF[k+1] = HpX_KFk1
        # if self.horizon>10:
        #     print("Fbar_Hktime=--- %s ms ---" % format(Fbar_Hktime/(horizon-1),'.2f'))
        #     print("HpX_kk1time=--- %s ms ---" % format(HpX_kk1time/(horizon-1),'.2f'))
        #     print("PH_k1time=--- %s ms ---" % format(PH_k1time/(horizon-1),'.2f'))
        #     print("SH_k1time=--- %s ms ---" % format(SH_k1time/(horizon-1),'.2f'))
        #     print("CH_k1time=--- %s ms ---" % format(CH_k1time/(horizon-1),'.2f'))
        #     print("TH_k1time=--- %s ms ---" % format(TH_k1time/(horizon-1),'.2f'))
        #     print("HpX_KFk1time=--- %s ms ---" % format(HpX_KFk1time/(horizon-1),'.2f'))

        """-------------------------Backward costate gradient--------------------------"""
        HpLAMBDA      = horizon*[np.zeros((self.n_para*self.n_xmhe,self.n_para))]
        HpLAMBDA_last = np.zeros((self.n_para*self.n_xmhe,self.n_para))  
        HpLAMBDA[-1]  = HpLAMBDA_last
        
        for k in range((horizon-1),0,-1):
            if k == horizon-1:
                HpLAMBDA_pre = S_H[k]@HpX_KF[k] + T_H[k]
            else:
                HpLAMBDA_pre = (I_nm + S_H[k]@C_H[k])@np.transpose(Fbar_H[k])@HpLAMBDA[k] + S_H[k]@HpX_KF[k] + T_H[k]
                
            HpLAMBDA[k-1] = HpLAMBDA_pre
        # if self.horizon>10:
        #     print("HpLAMBDA_time=--- %s ms ---" % format(HpLAMBDA_time/(horizon-1),'.2f'))

        """-------------------------Forward state gradient-----------------------------"""
        HpX_opt   = horizon*[np.zeros((self.n_para*self.n_xmhe,self.n_para))]
        for k in range(horizon):
            if k<horizon-1:
                HpX_optk = HpX_KF[k] + C_H[k]@np.transpose(Fbar_H[k])@HpLAMBDA[k]
                
            else:
                HpX_optk = HpX_KF[k]
            HpX_opt[k] = HpX_optk
        # if self.horizon>10:
        #     print("HpX_optk_time=--- %s ms ---" % format(HpX_optk_time/(horizon-1),'.2f'))
        
        return HpX_opt

    def loss_tracking(self, xpa, gt):
        loss_fn = Function('loss', [self.xa, self.x_t], [self.loss], ['xpa0', 'gt0'], ['lossf'])
        loss_track = loss_fn(xpa0=xpa, gt0=gt)['lossf'].full()
        return loss_track

    def loss_horizon(self, xmhe_traj, gt, N, time):
        loss_track = 0
        if time < N:
            horizon = time + 1
        else:
            horizon = N + 1
        for k in range(horizon):
            x_mhe = xmhe_traj[k, :]
            dloss_track = self.Kloss * self.loss_tracking(x_mhe, gt[len(gt)-horizon+k])
            loss_track +=  dloss_track
        return loss_track
        
    def ChainRule(self, gt, xmhe_traj, X_opt,weight_grad):
        # Define the gradient of loss w.r.t state
        Ddlds = jacobian(self.loss, self.xa)
        Ddlds_fn = Function('Ddlds', [self.xa, self.x_t], [Ddlds], ['xpa0', 'gt0'], ['dldsf'])
        # Initialize the parameter gradient
        dp = np.zeros((1, self.n_para))
        # Initialize the loss
        loss_track = 0
        # Positive coefficient in the loss
        for t in range(self.horizon):
            x_mhe = xmhe_traj[t, :]
            x_mhe = np.reshape(x_mhe, (self.n_xmhe, 1))
            dloss_track = self.Kloss * self.loss_tracking(x_mhe, gt[len(gt)-self.horizon+t])
            loss_track +=  dloss_track
            dlds =  self.Kloss * Ddlds_fn(xpa0=x_mhe, gt0=gt[len(gt)-self.horizon+t])['dldsf'].full()
            dxdp = X_opt[t]
            dp  += np.matmul(dlds, dxdp)
        Grad_dldp = dp@weight_grad
        return Grad_dldp, loss_track
    
    def ChainRule_2nd(self,gt,xmhe_traj,X_opt,HpX_opt,weight_grad,weight_hess,neural_grad,neural_hess):
        # Define the gradient of loss w.r.t state
        Ddlds = jacobian(self.loss, self.xa)
        Ddlds_fn = Function('Ddlds', [self.xa, self.x_t], [Ddlds], ['xpa0', 'gt0'], ['dldsf'])
        Hdlds = jacobian(Ddlds,self.xa)
        Hdlds_fn = Function('Hdlds', [self.xa], [Hdlds], ['xpa0'], ['Hdldsf'])
        # Initialize the parameter gradient and hessian
        dp   = np.zeros((1, self.n_para))
        Hxdp = np.zeros((self.n_para,self.n_para))
        Hdp  = np.zeros((self.n_para,self.n_para))
        # Initialize the loss
        loss_track = 0
        # Gradient and hessian in the loss, p: the weight parameters used directly in MHE, w: the parameterized p, n: the output of the neural network
        dpdn = weight_grad@neural_grad
        start_time = TM.time()
        hess_dpdn = np.transpose(np.reshape(weight_hess@neural_grad,(self.n_para, (self.n_para*self.n_neur))))@neural_grad + np.kron(weight_grad,np.identity(self.n_neur))@neural_hess
        hesstime = (TM.time() - start_time)
        # print("chainrule_hesstime=--- %s s ---" % format(hesstime,'.2f'))
        for t in range(self.horizon):
            x_mhe = xmhe_traj[t, :]
            x_mhe = np.reshape(x_mhe, (self.n_xmhe, 1))
            dloss_track = self.Kloss * self.loss_tracking(x_mhe, gt[len(gt)-self.horizon+t])
            loss_track +=  dloss_track
            dxdp = X_opt[t]
            hxdp = HpX_opt[t]
            dldx = self.Kloss * Ddlds_fn(xpa0=x_mhe, gt0=gt[len(gt)-self.horizon+t])['dldsf'].full()
            Hess_dldx = self.Kloss * Hdlds_fn(xpa0=x_mhe)['Hdldsf'].full()
            Hxdp += np.transpose(dxdp)@Hess_dldx@dxdp
            Hdp  += np.kron(dldx,np.identity(self.n_para))@hxdp
            # Hdp  += np.kron(np.identity(self.n_para),dldx)@hxdp # correct order in the Kronecker product will cause numerical issues such as complex eigenvalues in Line 625, see 'Discussion on Kronecker Product' PDF for details.
            dp   += dldx@dxdp
        Grad_dldn = dp@dpdn
        Hess_dldn = np.transpose((Hxdp + Hdp)@dpdn)@dpdn + np.kron(dp,np.identity(self.n_neur))@hess_dpdn
        # Hess_dldn = np.transpose(dpdn)@(Hxdp + Hdp)@dpdn + np.kron(np.identity(self.n_neur),dp)@hess_dpdn # will cause numerical issues such as complex eigenvalues in Line 625
        # Hess_dldn = np.transpose(dpdn)@(Hxdp + Hdp)@dpdn + np.kron(dp,np.identity(self.n_neur))@hess_dpdn # Even slightly modifying the first term does not affect the training results, showing superior robustness of our method.
        return Grad_dldn, Hess_dldn, loss_track








    
