import torch 
import scipy.linalg
import numpy as np
import itertools

from scipy.linalg import fractional_matrix_power

device = "cpu" if not torch.backends.cuda.is_built() else "cuda:0"

class Recorder:
    
    def __init__(self,m,k):
        
        self.m = m
        self.k = k
      
    # Function: record
    # Input:    self
    #           X (torch array)
    #           T (torch array)
    #           tEND (int)
    # Process:  input value into the object to be recorded. Also, record optimal values
    # Output:   none
    def record(self,name='Random'):
        
        if name in ['Random']:
        
            self.X = np.array([[]])
            self.T = np.array([[]])

            self.choices_agent = np.zeros([self.k,1]) 
            self.X_agent       = np.zeros([1,1])

            self.pred_agent    = np.zeros([self.k,1])
        
        elif name in ['Kalman_UCB','Kalman_Observer','Kalman_Greedy','UCB','LBSS','UBSS','HyperController','HyperBand']:
            
            self.X = torch.Tensor([[]]).to(device)
            self.T = torch.zeros([self.m,1]).to(device)

            self.choices_agent = torch.zeros([self.k,1]).to(device)
            self.X_agent       = torch.zeros([1,1]).to(device)

            self.pred_agent    = torch.zeros([self.k,1]).to(device)
        
# -------------------------------------------------------------------------------------------------------------------
        
# Function: interact
# Input:    self
#           record (object)
# Process:  interact with environment based on an inputted dataset from record
# Output:   none
def interact(agent,record,X_input,T_input,t_ready,iteration,mode='max'):
        
    if mode != 'max':
        accuracy = - np.float64(np.array(X_input))
    
    if agent.name in ['Kalman_UCB','Kalman_Observer','Kalman_Greedy','PB2','UCB','LBSS','UBSS']:
                
        record.X = torch.cat((record.X,torch.from_numpy(np.array([X_input])).unsqueeze(1).to(device).double()),dim=1)
               
        agent.learn(record,iteration,agent.prev_action)
        agent.predict(record,iteration)
            
        if iteration % t_ready == 0:
            action = agent.make_choice(record,iteration)
        else:
            action = agent.prev_action

    if agent.name in ['HyperController']:
                
        record.X = torch.cat((record.X,torch.from_numpy(np.array([X_input])).unsqueeze(1).to(device).double()),dim=1)
               
        agent.learn(record,iteration,agent.prev_action)
        agent.predict(record,iteration)
            
        if iteration % t_ready == 0:
            action = agent.make_choice(record,iteration,False)
        else:
            action = agent.make_choice(record,iteration,True)
            
    elif agent.name in ['HyperBand']:

        record.X = torch.cat((record.X,torch.from_numpy(np.array([X_input])).unsqueeze(1).to(device).double()),dim=1)
        action   = agent.make_choice(record,iteration)

    return action

# -------------------------------------------------------------------------------------------------------------------
    
class Feedback_Kalman():
    
    def __init__(self,k,hyperparameters,name):
    
        self.k               = k
        self.hyperparameters = hyperparameters 

        if name == 'Kalman_UCB':
            self.name = 'Kalman_UCB'
        elif name == 'Kalman_Observer':
            self.name = 'Kalman_Observer'
        elif name == 'Kalman_Greedy':
            self.name = 'Kalman_Greedy'
        
        self.reset_values()
    
    def reset_values(self):

        self.P = torch.eye(self.k,dtype=torch.float64).to(device)
        self.A = 0.9*torch.eye(self.k,dtype=torch.float64).to(device)
        self.Q = torch.zeros([self.k,self.k],dtype=torch.float64).to(device)
        
        for ii in range(self.k):
            for jj in range(self.k):
                if ii != jj:
                    dij = torch.Tensor([self.hyperparameters[ii]-self.hyperparameters[jj]])
                    self.Q[ii,jj] = -torch.exp(-torch.pow(dij,2)/2)

        for ii in range(self.k): 
            self.Q[ii,ii] = 1-self.Q[ii,:].sum()

        self.Q = torch.linalg.inv(self.Q).to(device)
        self.c = torch.zeros([self.k,1],dtype=torch.float64).to(device)
        self.c[0] = 1
        self.K = self.P@self.c@torch.linalg.inv(self.c.T@self.P@self.c+1)
        
        self.actions      = torch.Tensor([0]).to(device)
        self.prev_action  = 0
        self.X_pred       = torch.zeros([self.k],dtype=torch.float64).to(device)
        self.perturbation = torch.zeros([self.k],dtype=torch.float64).to(device)
        
    def learn(self,record,t,a):
        
        X = record.X.to(device)

        self.c = torch.zeros([self.k,1],dtype=torch.float64).to(device)
        self.c[self.prev_action] = 1

        self.X_pred = self.A@self.X_pred + self.A@self.K@(X[:,t]-self.c.T@self.X_pred)
        self.P      = self.A@self.P@self.A.T + self.Q - self.A@self.P@self.c@torch.linalg.inv(self.c.T@self.P@self.c+1)@self.c.T@self.P@self.A.T
        self.K      = self.P@self.c@torch.linalg.inv(self.c.T@self.P@self.c+1)

        for aa in range(self.k):
            self.c     = torch.zeros([self.k,1],dtype=torch.float64).to(device)
            self.c[aa] = 1
            if self.name == 'Kalman_UCB':
                self.perturbation[aa] = torch.sqrt(self.c.T@self.P@self.c).squeeze()
            elif self.name == 'Kalman_Observer':
                self.perturbation[aa] = torch.sqrt(torch.trace(self.A@self.P@self.c@torch.linalg.inv(self.c.T@self.P@self.c+1)@self.c.T@self.P@self.A.T)).squeeze()
            elif self.name == 'Kalman_Greedy':
                self.perturbation[aa] = 0
            else:
                raise Exception("Algorithm not available")
    
    def predict(self,record,t):

        X = record.X.to(device)
            
    def make_choice(self,record,t):

        self.prev_action = torch.argmax(self.X_pred.squeeze()+self.perturbation.squeeze())
        self.actions     = torch.cat((self.actions,torch.Tensor([self.prev_action]).to(device)),dim=0)
        
        return self.prev_action
        
class LBSS():
    
    def __init__(self,s,k,d):
    
        self.s = s
        self.k = k
        self.d = d
        
        self.R   = 10
        self.B_c = 1
        self.B_G = 10
        
        self.delta     = 0.1
        self.log_delta = torch.Tensor([np.log(1/self.delta)]).to(device)
        self.delta     = torch.Tensor([self.delta]).to(device)
        
        self.name = 'LBSS'
        
        self.reset_values()
    
    def reset_values(self):
        
        self.code_V = {}
        self.code_B = {}
        self.code_G = {}
        self.code_H = {}
        self.code_N = {}
        self.code_T = {}
        
        self.code_words = [''.join(list([str(jj) for jj in ii])) for ii in itertools.product(list(range(self.k)), repeat=self.s)]
        self.encoders   = [''.join(list([str(jj) for jj in ii])) for ii in itertools.product(list(range(self.k)), repeat=self.s+1)]

        for encoder in self.encoders:
            self.code_V[encoder] = torch.eye(self.s,dtype=torch.float64).to(device)
            self.code_B[encoder] = torch.zeros([self.s,1],dtype=torch.float64).to(device)
            self.code_G[encoder] = torch.zeros([self.s,1],dtype=torch.float64).to(device)
            self.code_N[encoder] = torch.Tensor([0]).to(device)
            self.code_T[encoder] = 0

        for code in self.code_words:
            self.code_H[code] = torch.zeros([self.k,self.s],dtype=torch.float64).to(device)
            
        self.actions     = torch.Tensor([0]).to(device)
        self.prev_action = 0
        self.X_pred      = torch.zeros([self.k]).to(device)
        self.code        = self.code_words[0]
        
        self.perturbation = torch.zeros([self.k]).to(device)
        
    def learn(self,record,t,a):
        
        X = record.X.to(device)
        
        if t-self.s-1 >= 0:
            
            a_encoder = ''.join([str(int(a)) for a in self.actions[t-self.s-1:t].tolist()])
            
            if len(a_encoder) > self.s:
                self.code_V[a_encoder] = self.code_V[a_encoder] + X[:,t-self.s-1:t-1].T@X[:,t-self.s-1:t-1]
                self.code_B[a_encoder] = self.code_B[a_encoder] + (X[:,t-1:t].T@X[:,t-self.s-1:t-1]).T
                self.code_G[a_encoder] = torch.linalg.solve(self.code_V[a_encoder],self.code_B[a_encoder])
                
                self.code_T[a_encoder] += 1

                V_inv = torch.linalg.inv(self.code_V[a_encoder])
                
                a    = int(a_encoder[0])
                code = str(a_encoder[1:])

                self.code_H[code][a,:] = self.code_G[a_encoder].squeeze()

                e_temp_1 = torch.sqrt(torch.det(self.code_V[a_encoder]))
                e_temp_2 = torch.sqrt(torch.det(torch.eye(self.s).to(device)))
                e_temp_3 = torch.sqrt(2*self.R*(self.log_delta + torch.log(e_temp_1/e_temp_2)))

                b_temp_1 = torch.Tensor([torch.sqrt(torch.trace(V_inv))*self.B_G]).to(device)
                b_temp_2 = torch.sqrt(torch.Tensor([self.code_T[a_encoder]]).to(device))*torch.sqrt(torch.Tensor([self.B_c*self.R]).to(device))*(1/self.delta)*torch.sqrt(torch.Tensor([torch.trace(torch.eye(self.s).to(device)-V_inv)]).to(device)) + b_temp_1

                self.perturbation[a] = (e_temp_3+b_temp_2)*torch.sqrt(X[:,t-self.s:t]@V_inv@X[:,t-self.s:t].T)

                self.code = code
            
    def predict(self,record,t):

        X = record.X.to(device)
        
        if t-self.s >= 0:

            self.X_pred = self.code_H[self.code]@X[:,t-self.s:t].T
            
    def make_choice(self,record,t):
        
        if t-self.s >= 0:
            self.prev_action = torch.argmax(self.X_pred.squeeze()+self.perturbation.squeeze())
        else:
            self.prev_action = torch.randint(self.k,(1,))[0]
        
        self.actions = torch.cat((self.actions,torch.Tensor([self.prev_action]).to(device)),dim=0)
        
        return self.prev_action

class UBSS():
    
    def __init__(self,s,k,d):
    
        self.s = s
        self.k = k
        self.d = d
        
        self.R   = 10
        self.B_c = 1
        self.B_G = 10
        
        self.delta     = 0.1
        self.log_delta = torch.Tensor([np.log(1/self.delta)]).to(device)
        self.delta     = torch.Tensor([self.delta]).to(device)
        
        self.name = 'LBSS'
        
        self.reset_values()
    
    def reset_values(self):
        
        self.code_V = {}
        self.code_B = {}
        self.code_G = {}
        self.code_H = {}
        self.code_N = {}
        self.code_T = {}
        
        self.code_words = [''.join(list([str(jj) for jj in ii])) for ii in itertools.product(list(range(self.k)), repeat=self.s)]
        self.encoders   = [''.join(list([str(jj) for jj in ii])) for ii in itertools.product(list(range(self.k)), repeat=self.s+1)]

        for encoder in self.encoders:
            self.code_V[encoder] = torch.eye(self.s,dtype=torch.float64).to(device)
            self.code_B[encoder] = torch.zeros([self.s,1],dtype=torch.float64).to(device)
            self.code_G[encoder] = torch.zeros([self.s,1],dtype=torch.float64).to(device)
            self.code_N[encoder] = torch.Tensor([0]).to(device)
            self.code_T[encoder] = 10000

        for code in self.code_words:
            self.code_H[code] = torch.zeros([self.k,self.s],dtype=torch.float64).to(device)
            
        self.actions     = torch.Tensor([0]).to(device)
        self.prev_action = 0
        self.X_pred      = torch.zeros([self.k]).to(device)
        self.code        = self.code_words[0]
        
        self.perturbation = torch.zeros([self.k]).to(device)
        
    def learn(self,record,t,a):
        
        X = record.X.to(device)
        
        if t-self.s-1 >= 0:
            
            a_encoder = ''.join([str(int(a)) for a in self.actions[t-self.s-1:t].tolist()])
            
            if len(a_encoder) > self.s:
                self.code_V[a_encoder] = self.code_V[a_encoder] + X[:,t-self.s-1:t-1].T@X[:,t-self.s-1:t-1]
                self.code_B[a_encoder] = self.code_B[a_encoder] + (X[:,t-1:t].T@X[:,t-self.s-1:t-1]).T
                self.code_G[a_encoder] = torch.linalg.solve(self.code_V[a_encoder],self.code_B[a_encoder])
                
                self.code_T[a_encoder] += 1

                V_inv = torch.linalg.inv(self.code_V[a_encoder])
                
                a    = int(a_encoder[0])
                code = str(a_encoder[1:])

                self.code_H[code][a,:] = self.code_G[a_encoder].squeeze()

                e_temp_1 = torch.sqrt(torch.det(self.code_V[a_encoder]))
                e_temp_2 = torch.sqrt(torch.det(torch.eye(self.s).to(device)))
                e_temp_3 = torch.sqrt(2*self.R*(self.log_delta + torch.log(e_temp_1/e_temp_2)))

                b_temp_1 = torch.Tensor([torch.sqrt(torch.trace(V_inv))*self.B_G]).to(device)
                b_temp_2 = torch.sqrt(torch.Tensor([self.code_T[a_encoder]]).to(device))*torch.sqrt(torch.Tensor([self.B_c*self.R]).to(device)*torch.log(torch.Tensor([self.code_T[a_encoder]]).to(device)/self.delta))*torch.sqrt(torch.Tensor([torch.trace(torch.eye(self.s).to(device)-V_inv)]).to(device)) + b_temp_1

                self.perturbation[a] = (e_temp_3+b_temp_2)*torch.sqrt(X[:,t-self.s:t]@V_inv@X[:,t-self.s:t].T)

                self.code = code
            
    def predict(self,record,t):

        X = record.X.to(device)
        
        if t-self.s >= 0:

            self.X_pred = self.code_H[self.code]@X[:,t-self.s:t].T
            
    def make_choice(self,record,t):

        if t-self.s >= 0:
            self.prev_action = torch.argmax(self.X_pred.squeeze()+self.perturbation.squeeze())
        else:
            self.prev_action = torch.randint(self.k,(1,))[0]
        
        self.actions = torch.cat((self.actions,torch.Tensor([self.prev_action]).to(device)),dim=0)
        
        return self.prev_action

# HyperController --------------------------------------------------------------------------------------------

class HyperController():
    
    def __init__(self,s,k,d):
    
        self.s = s
        self.k = k
        self.d = d
        
        self.R   = 10
        self.B_c = 1
        self.B_G = 10

        self.name   = 'HyperController'
        self.device = "cpu" if not torch.backends.cuda.is_built() else "cuda:0"
    
        self.delta     = 0.1
        self.log_delta = torch.Tensor([np.log(1/self.delta)]).to(self.device)
        self.delta     = torch.Tensor([self.delta]).to(self.device)
        
        self.reset_values()
    
    def reset_values(self):
        
        self.code_V = {}
        self.code_B = {}
        self.code_G = {}
        self.code_H = {}
        self.code_N = {}
        self.code_T = {}
        
        self.code_words = [''.join(list([str(jj) for jj in ii])) for ii in itertools.product(list(range(self.k)), repeat=self.s)]
        self.encoders   = [''.join(list([str(jj) for jj in ii])) for ii in itertools.product(list(range(self.k)), repeat=self.s+1)]

        for encoder in self.encoders:
            self.code_V[encoder] = torch.eye(self.s,dtype=torch.float64).to(self.device)
            self.code_B[encoder] = torch.zeros([self.s,1],dtype=torch.float64).to(self.device)
            self.code_G[encoder] = torch.zeros([self.s,1],dtype=torch.float64).to(self.device)
            self.code_N[encoder] = torch.Tensor([0]).to(self.device)
            self.code_T[encoder] = 0

        for code in self.code_words:
            self.code_H[code] = torch.zeros([self.k,self.s],dtype=torch.float64).to(self.device)
            
        self.actions     = torch.Tensor([0]).to(self.device)
        self.prev_action = 0
        self.X_pred      = torch.zeros([self.k]).to(self.device)
        self.code        = self.code_words[0]
        
        self.perturbation = torch.zeros([self.k]).to(self.device)#1/(float(1e-8)+torch.zeros([self.k]).to(self.device))
        
    def learn(self,record,t,a):
        
        X = record.X.to(self.device)
        if t-self.s-1 >= 0:
            
            a_encoder = ''.join([str(int(a)) for a in self.actions[t-self.s-1:t].tolist()])

            if len(a_encoder) > self.s:
                self.code_V[a_encoder] = self.code_V[a_encoder] + X[:,t-self.s-1:t-1].T@X[:,t-self.s-1:t-1]
                self.code_B[a_encoder] = self.code_B[a_encoder] + (X[:,t-1:t].T@X[:,t-self.s-1:t-1]).T
                self.code_G[a_encoder] = torch.linalg.solve(self.code_V[a_encoder],self.code_B[a_encoder])

                self.code_T[a_encoder] += 1

                V_inv = torch.linalg.inv(self.code_V[a_encoder])
                
                a    = int(a_encoder[0])
                code = str(a_encoder[1:])

                self.code_H[code][a,:] = self.code_G[a_encoder].squeeze()

                # e_temp_1 = torch.sqrt(torch.det(self.code_V[a_encoder]))
                # e_temp_2 = torch.sqrt(torch.det(torch.eye(self.s).to(self.device)))
                # e_temp_3 = torch.sqrt(2*self.R*(self.log_delta + torch.log(e_temp_1/e_temp_2)))

                # b_temp_1 = torch.Tensor([torch.sqrt(torch.trace(V_inv))*self.B_G]).to(self.device)
                # b_temp_2 = b_temp_1 + torch.sqrt(torch.Tensor([self.code_T[a_encoder]]).to(self.device))*torch.sqrt(torch.Tensor([self.B_c*self.R]).to(self.device)*torch.log(torch.Tensor([self.code_T[a_encoder]]).to(self.device)/self.delta))*torch.sqrt(torch.Tensor([torch.trace(torch.eye(self.s).to(self.device)-V_inv)]).to(self.device))

                self.perturbation[a] = 0#(e_temp_3+b_temp_2)*torch.sqrt(X[:,t-self.s:t]@V_inv@X[:,t-self.s:t].T)

                self.code = code
            
    def predict(self,record,t):

        X = record.X.to(self.device)
        
        if t-self.s >= 0:

            self.X_pred = self.code_H[self.code]@X[:,t-self.s:t].T
            
    def make_choice(self,record,t,skip):
        
        if skip == False:
            if t-self.s >= 0:
                self.prev_action = torch.argmax(self.X_pred.squeeze()+self.perturbation.squeeze())
            else:
                self.prev_action = torch.randint(self.k,(1,))[0]
        
        self.actions = torch.cat((self.actions,torch.Tensor([self.prev_action]).to(self.device)),dim=0)
        
        return self.prev_action
        
# Comparison Codes ----------------------------------------------------------------------------------------------------------
class UCB:
    
    def __init__(self,window_size,k):
               
        self.k                  = k
        self.R                  = 10#R
        self.window_size        = window_size
        self.tEND               = 1000000
        if window_size == 0:
            self.use_sliding_window = False
            self.name = 'UCB'
        else:
            self.use_sliding_window = True
            self.name = 'SW-UCB'

        self.decay_rate = 1.0
        
        self.reset_values()
        
    # Function: reset_values
    # Input:    self
    # Process:  (re)set model 
    # Output:   none
    def reset_values(self):
       
        self.S   = torch.zeros([self.k,1],dtype=torch.float64).to(device)
        self.T   = torch.zeros([self.k,1],dtype=torch.float64).to(device)
        self.mu  = torch.zeros([self.k,1],dtype=torch.float64).to(device)
        self.UCB = 1/(torch.zeros([self.k,1],dtype=torch.float64).to(device)+float(1e-100))

        self.reward_seq     = torch.zeros([self.k,self.tEND],dtype=torch.float64).to(device) #np.zeros()
        self.action_seq     = torch.zeros([self.k,self.tEND],dtype=torch.float64).to(device) #np.zeros([self.k,self.tEND])
        self.X_pred         = torch.zeros([self.k],dtype=torch.float64).to(device) #np.zeros([self.k])
        self.perturb_signal = torch.zeros([self.k],dtype=torch.float64).to(device) #np.zeros([self.k])
        
        self.prev_action = 0
        
    # Function: learn
    # Input:    self
    #           record (object)
    #           t (int)
    #           a (int)
    # Process:  learn the matrix G for reward prediction
    # Output:   none
    def learn(self,record,t,a):

        X = record.X.to(device).T
        t = X.shape[0]
        X = X.T
        if t > 0:
            if self.use_sliding_window == False or t < self.window_size:
    
                self.T[a]  = self.T[a] + 1
                self.S[a]  = self.S[a] + X[:,t-1:t].squeeze()
                self.mu[a] = self.S[a]/self.T[a]
                
                self.UCB[a] = self.mu[a] + torch.sqrt(2*self.R*torch.log(torch.Tensor([0.1])).to(device)/self.T[a])
    
            else:
    
                self.reward_seq[a,t] = X[:,t-1:t].squeeze()
                self.action_seq[a,t] = 1
    
                for aa in range(self.k):
                    
                    self.mu[aa]  = self.reward_seq[aa,t-self.window_size:t].sum()
                    self.T[aa]   = self.action_seq[aa,t-self.window_size:t].sum()
                    if self.T[aa] > 0:
                        self.UCB[aa] = self.mu[aa]/self.T[aa] + torch.sqrt(2*self.R*torch.log(torch.Tensor([0.1])).to(device)/self.T[aa])

    # Function: predict
    # Input:    self
    #           record (object)
    #           t (int)
    # Process:  predict the reward for each action a in A
    # Output:   np.matmul(self.G,Z).squeeze() (np array) or np.zeros([self.k,1]).squeeze()
    def predict(self,record,t):
        
        self.X_pred = self.mu
        return self.X_pred
        
    # Function: make_choice
    # Input:    self
    #           record (object)
    #           t (int)
    # Process:  make a choice based on the reward prediction
    # Output:   a (int)
    def make_choice(self,record,t):
        
        self.prev_action = torch.argmax(self.UCB)
        
        return self.prev_action

class HyperBand:
    
    def __init__(self,R,hyperparameter_bounds,eta=3):
               
        self.eta                   = eta
        self.R                     = R
        self.hyperparameter_bounds = hyperparameter_bounds

        self.name = 'HyperBand'
        
        self.reset_values()
        
    # Function: reset_values
    # Input:    self
    # Process:  (re)set model 
    # Output:   none
    def reset_values(self):

        # Initialize Values
        self.s_max      = np.floor(np.log(self.R)/np.log(self.eta))
        self.B          = (self.s_max+1)*self.R
        
        self.s          = self.s_max
        self.i          = 0
        self.iter       = 0
        self.sample_loc = 0

        # Call whenever get_hyperparameter_configuration()
        self.n = int(np.ceil((self.B/self.R)*(np.power(self.eta,self.s)/(self.s+1))))
        self.r = self.R*np.power(self.eta,-self.s)
        self.get_hyperparameter_configuration()

        # Call whenever get_hyperparameter_configuration()
        self.ni = int(np.floor(self.n*np.power(self.eta,-self.i)))
        self.ri = self.r*np.power(self.eta,self.i)

        self.STOP = False
        
    # Function: get_hyperparameter_configuration
    # Input:    self
    #           record (object)
    #           t (int)
    #           a (int)
    # Process:  return a random set of hyperparameters
    # Output:   none
    def get_hyperparameter_configuration(self):
        
        self.T = {}
        self.L = torch.zeros([self.n],dtype=torch.float64).to(device)
        for key in self.hyperparameter_bounds.keys():
            self.T[key] = (self.hyperparameter_bounds[key][1]-self.hyperparameter_bounds[key][0])*np.random.rand(self.n)+self.hyperparameter_bounds[key][0]
            
    # Function: run_then_return_val_loss
    # Input:    self
    #           record (object)
    #           t (int)
    #           a (int)
    # Process:  iterate till out of rounds
    # Output:   none
    def run_then_return_val_loss(self,record,t):
        
        X = record.X.to(device).T
        t = X.shape[0]
        X = X.T
        X = X[:,t-1:t].squeeze()
        self.L[self.sample_loc] += X/self.ri
        
    # Function: run_then_return_val_loss
    # Input:    self
    #           record (object)
    #           t (int)
    #           a (int)
    # Process:  iterate till out of rounds
    # Output:   none
    def top_k(self):

        locs   = torch.argsort(self.L,descending=True)
        T_temp = {}
        for key in self.T.keys():
            T_temp[key] = np.zeros([np.floor(self.ni/self.eta)])
            for ii in range(np.floor(self.ni/self.eta)):
                T_temp[key][ii] = self.T[key][locs[ii]]
        self.T = T_temp
        self.L = torch.zeros([self.n],dtype=torch.float64).to(device)

    # Function: check_iterations
    # Input:    self
    #           record (object)
    #           t (int)
    # Process:  There are 4 loops 
    #           (1) Loop through sample          (self.iter)
    #           (2) Loop through configurations  (self.sample_loc)
    #           (3) Loop through i loop          (self.i)
    #           (4) Loop through s loop          (self.s)
    # Output:   none
    def check_iterations(self,record,t):

        # Check Loop (1)
        if self.iter > self.ri:
            self.iter        = 0
            self.sample_loc += 1
        # Check Loop (2)
        if self.sample_loc >= self.L.shape[0]:
            self.top_k(record,t)
            self.i += 1
            self.ni = int(np.floor(self.n*np.power(self.eta,-self.i)))
            self.ri = self.r*np.power(self.eta,self.i)
        # Check Loop (3)
        if self.i >= self.s:
            self.s += -1
            self.n = int(np.ceil((self.B/self.R)*(np.power(self.eta,self.s)/(self.s+1))))
            self.r = self.R*np.power(self.eta,-self.s)
            self.get_hyperparameter_configuration()
        # Check Loop (4)
        if self.s < 0:
            self.STOP = True
        
    # Function: make_choice
    # Input:    self
    #           record (object)
    #           t (int)
    # Process:  make a choice based on the reward prediction
    # Output:   a (int)
    def make_choice(self,record,t):

        if self.STOP == False:
            self.run_then_return_val_loss(record,t)
            self.iter += 1
            self.check_iterations(record,t)
        T_temp = {}
        for key in self.T.keys():
            T_temp[key] = self.T[key][self.sample_loc]
        return T_temp
                    
class Random_Agent():
    
    def __init__(self,s,k,d):
    
        self.s = s
        self.k = k
        self.d = d
       
        self.name = 'Random'
        self.prev_action = 0
        
        self.reset_values()
    
    def reset_values(self):
        
        pass
        
    def learn(self,record,t,a):
        
        pass
            
    def predict(self,record,t):

        pass

    def make_choice(self,record,t):
        
        self.prev_action = torch.randint(self.k,(1,))[0].to(device)
        
        return self.prev_action
