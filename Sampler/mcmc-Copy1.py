from numpy import random,exp
import numpy as np

class Sampler:
    def __init__(self,loglkl,init_state,step_size,cosmo=None):
        self.cosmo=cosmo
        self.paramN = len(step_size)
        self.loglkl = loglkl
        self.step_size = step_size
        #self.curr_state = prior()
        self.init_state = init_state
        self.init()

    def init(self):
        self.curr_state = np.array(self.get_state(self.init_state))
        self.curr_lk = exp(self.loglkl(self.curr_state,cosmo=self.cosmo))

    def get_chain(self,N,method='MH'):
        self.init()
        if method == 'MH':
            return self.metropolis_hastings(N)
        else:
            raise ValueError('No other method implemented yet')

    def get_state(self,distribution,*args):
        temp = distribution(*args) 
        return temp
        
    def proposal_distribution(self):
        _ = np.array([self.get_state(random.normal,0,self.step_size[i]) for i in range(self.paramN)])
        return _
        
        
        
    def metropolis_hastings(self,N):
        def mcmc_updater():
            #print(self.curr_state,self.proposal_distribution())
            # propose_state = self.curr_state + self.proposal_distribution()
            # propose_lk = exp(self.loglkl(propose_state,cosmo=self.cosmo))
            # accept_crit = propose_lk/self.curr_lk
    
            # alpha = random.uniform(0,1)
            # if alpha<accept_crit:
            #     self.curr_state = np.array(propose_state)
            #     self.curr_lk = propose_lk

            
            propose_state = self.curr_state + self.proposal_distribution()
            propose_lk = exp(self.loglkl(propose_state,cosmo=self.cosmo))
            accept_crit = propose_lk/self.curr_lk
            alpha = random.uniform(0,1)

            while alpha>accept_crit:
                propose_state = self.curr_state + self.proposal_distribution()
                propose_lk = exp(self.loglkl(propose_state,cosmo=self.cosmo))
                accept_crit = propose_lk/self.curr_lk
                alpha = random.uniform(0,1)
            self.curr_state = np.array(propose_state)
            self.curr_lk = propose_lk
                

        sample = np.empty(shape=(N+1,self.paramN))
        sample[0]=self.curr_state
        for i in range(N):
            if self.cosmo:
                # this is because there is "CosmoComputationError" when calculating cosmo, which undefined in python; thus I want catch it in comos caluclation otherwise "except" will catch everthing
                try:
                    mcmc_updater()
                    sample[i+1]=self.curr_state
                except:
                    
                    sample[i+1]=self.curr_state
                    print('bad value',self.curr_state)
            else:
                mcmc_updater()
                sample[i+1]=self.curr_state
                
                
            
        return np.array(sample)

