import numpy as np
import random as rr
import matplotlib.pyplot as py
from utils import Rollout
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from matplotlib.patches import Arrow



class GridWorld:

    def __init__(self,grid,initial,final):

        ## grid - is gridworld to be tested on: (0,0) should be upper left, (x,y) should be lower right
        ## initial - initial state in form (row,col) of grid world at which to start
        ## final - final states, list of final states

        
        self.rewards=grid
        self.moves=[1,2,3,4] # 1 - UP, 2 - DOWN, 3 - LEFT, 4 - RIGHT
        self.state=initial
        self.initial=initial
        self.final={}
        for i in final:
            self.final[i]=1

        x,y=np.shape(grid)
        self.x=x-1
        self.y=y-1
        pos={}
        n=0
        for i in range(self.x+1):
            for j in range(self.y+1):
                pos[(i,j)]=n
                n+=1
        self.pos=pos
        
    def reinitialize(self):
        #reinitialize MDP
        self.state=self.initial
        
    def step(self,action):

        # steps the MDP to the next state (note: uncertainty occurs in this func)
        
        move=[0,1,-1]
        rand=rr.random()
        if rand>0.75:
            action=rr.choice([1,2,3,4])

        if action in self.moves:
            if action==1 and self.state[0]==0:
                r=self.rewards[self.state[0],self.state[1]]
                final=self.final.get(self.state,0)

            elif action==2 and self.state[0]==self.x:
                r=self.rewards[self.state[0],self.state[1]]
                final=self.final.get(self.state,0)

            elif action==3 and self.state[1]==0:
                r=self.rewards[self.state[0],self.state[1]]
                final=self.final.get(self.state,0)

            elif action==4 and self.state[1]==self.y:
                r=self.rewards[self.state[0],self.state[1]]
                final=self.final.get(self.state,0)

            else:
            
                if action in [1,2]:
                    newrow=self.state[0]-move[action]
                    self.state=(newrow,self.state[1])

                    r=self.rewards[self.state[0],self.state[1]]
                    final=self.final.get(self.state,0)

                else:
                    newcol=self.state[1]-move[action-2]
                    self.state=(self.state[0],newcol)

                    r=self.rewards[self.state[0],self.state[1]]
                    final=self.final.get(self.state,0)

        return (self.state,r,final,action)
            
    def viewTrajectory(self,tau,policy,speed=0.01):

        #visulatization of rollout tau (can edit speed to make it faster)

        moves={1:[0,1],2:[0,-1],3:[-1,0],4:[1,0],0:[0,0]}

        py.ion()

        fig=py.figure()
        gca=fig.gca()
        py.xlim([0,self.y+1])
        py.ylim([0,self.x+1])


        mini=-1*max(abs(np.floor(np.min(self.rewards))),abs(np.floor(np.max(self.rewards))))
        maxi=max(abs(np.floor(np.min(self.rewards))),abs(np.floor(np.max(self.rewards))))
        color=list(py.cm.coolwarm(np.linspace(0,1,(maxi-mini+1))))
        
        
        locs={}
        pos={}
        for i in range(self.x+1):
            for j in range(self.y+1):
                pos[(i,j)]=[j+0.5,self.x-i+0.5]
                locs[self.pos[(i,j)]]=[j+0.5,self.x-i+0.5]

                if self.final.get((i,j),0)==1:
                    sh='/'
                else:
                    sh=''

                c=color[int(np.floor(self.rewards[i,j])-mini)]

                
                gca.add_patch(Rectangle((j,self.x-i), 1, 1, facecolor=c,hatch=sh))


        
        for i in range(len(self.pos.keys())):
            for k in range(1,len(self.moves)+1):
                gca.add_patch(Arrow(locs[i][0],locs[i][1],moves[k][0]*policy[i,k-1]*0.5,moves[k][1]*policy[i,k-1]*0.5,width=0.05,facecolor='black'))



        for i in range(len(tau[:,:2])):
            location=pos[(tau[i,0],tau[i,1])]
            ad=gca.add_patch(Circle(pos[(tau[i,0],tau[i,1])], radius=0.1, facecolor='red'))
            ad1=gca.add_patch(Arrow(location[0],location[1],moves[tau[i,2]][0]*0.2,moves[tau[i,2]][1]*0.2,width=0.05,facecolor='black'))
            py.pause(speed)
            ad.remove()
            ad1.remove()

        py.ioff()
        
        
        py.clf()
        py.cla()
        py.close()

    def viewPolicy(self,policy):

        #visualization of current policy

        moves={1:[0,1],2:[0,-1],3:[-1,0],4:[1,0],0:[0,0]}

        fig=py.figure()
        gca=fig.gca()
        py.xlim([0,self.y+1])
        py.ylim([0,self.x+1])
        



        mini=-1*max(abs(np.floor(np.min(self.rewards))),abs(np.floor(np.max(self.rewards))))
        maxi=max(abs(np.floor(np.min(self.rewards))),abs(np.floor(np.max(self.rewards))))
        color=list(py.cm.coolwarm(np.linspace(0,1,(maxi-mini+1))))
        
        
        locs={}
        pos={}
        for i in range(self.x+1):
            for j in range(self.y+1):
                pos[(i,j)]=[j+0.5,self.x-i+0.5]
                locs[self.pos[(i,j)]]=[j+0.5,self.x-i+0.5]

                if self.final.get((i,j),0)==1:
                    sh='/'
                else:
                    sh=''

                c=color[int(np.floor(self.rewards[i,j])-mini)]

                
                gca.add_patch(Rectangle((j,self.x-i), 1, 1, facecolor=c,hatch=sh))


                
        
        for i in range(len(self.pos.keys())):
            for k in range(1,len(self.moves)+1):
                gca.add_patch(Arrow(locs[i][0],locs[i][1],moves[k][0]*policy[i,k-1]*0.5,moves[k][1]*policy[i,k-1]*0.5,width=0.05,facecolor='black'))

        py.show()
        return



class Policy:

    def __init__(self,MDP,alpha=0.1,discount=0.9,initial=0):

        ## initial  - initial set of theta
        ##           (should be a nx4 matrix with n=number of states in GridWorld)
        ## MDP      - The Gridworld object this policy is for
        ## alpha    - step size for gradient descent method
        ## discount - discount for rewards

        if type(initial)!=type(0):
            self.theta=initial
        else:
            self.theta=np.random.rand((MDP.x+1)*(MDP.y+1),4)-0.5
            
            
        self.policy=self.getPolicy()
        self.MDP=MDP
        self.discount=discount
        self.alpha=alpha
        self.pos=MDP.pos
        self.initial=self.theta

    
    def changeGradientStepSize(self,alpha):
        self.alpha=alpha

    def changeDiscount(self,dis):
        self.discount=dis

    def getPolicy(self):

        ## calculates policy from theta (soft-max)
        
        policy=np.zeros(np.shape(self.theta))

        x,y=np.shape(self.theta)
        for i in range(x):
            sumi=0
            for j in range(y):
                policy[i,j]=np.exp(self.theta[i,j])
                sumi+=policy[i,j]

            policy[i,:]=policy[i,:]/sumi

        return policy
            
    def expectedReward(self,taus):

        # calculates a sample mean of the rewards for a collection of trajectories taus
        
        r=0;
        for traj in taus:
            r+=self.calcReward(traj,True)

        return r/len(taus)
        
        
    def Reinforce(self,n,epsilon,view=False,N=200,gradform1=False,baseline=True):

        #reinforce algorithm:
        # n        - number of rollouts
        # epsilon  - threshold value for change in sample mean reward
        # view     - True if want to see sample trajectory every 10 iterations
        # N        - N number of iterations (in case takes too long to converge to epsilon)

        theta=self.theta
        oldreward=1e20
        diff=10*epsilon
        j=0
        avgreward=[]
        
        while diff>epsilon and j<=N:
            j+=1
            taus=[]
            for i in range(n):
                traj=Rollout(self.MDP,self.policy)
                taus.append(traj)
            
            if j%10==0 and view:
                w=rr.randint(0, n-1)
                self.viewTrajectory(taus[w])
            
            grad=1.0/n
            sumi=0
            if gradform1 and baseline:
                b=self.calcBaseline(taus)
            else:
                b=0
            
            for traj in taus:    
                gradlog=self.calcGrad(traj,gradform1,b)
                sumi+=gradlog

            grad=sumi*grad
            theta=theta+self.alpha*grad

            self.theta=theta
            
            policy=self.getPolicy()
            
            self.policy=policy

            temp=self.expectedReward(taus)
            diff=abs(temp-oldreward)
            oldreward=temp
            avgreward.append(oldreward)
            print str(j)+': '+str(diff)

        return avgreward
            
    def calcGrad(self,tau,gradform,b=0):

        #calculates the gradient of log pi_theta(a|s) for the given trajectory
        #tau - trajectory from rollout
        

        
        totalReward=self.calcReward(tau,gradform)

        sumi=0

        
        for i in range(len(tau[:-1,0:3]-1)):
            sa=tau[i,0:3]
            grad=np.zeros(np.shape(self.theta))
            index=self.pos[(sa[0],sa[1])]

            for act in range(len(self.MDP.moves)):
                grad[index,act]=np.exp(self.theta[index,act])
            
            denom=np.sum(grad[index,:])

            grad[index,:]=-1*grad[index,:]/denom
            
            
            grad[index,sa[2]-1]=1+grad[index,sa[2]-1]
            
            if not gradform:
                sumi+=grad*sum(totalReward[i:])
            else:
                sumi+=grad

        
        if gradform:
            sumi=np.multiply(sumi,totalReward-b)
            
            
        return sumi


    def calcBaseline(self,taus):

        #calculates the gradient of log pi_theta(a|s) for the given trajectory
        #tau - trajectory from rollout
        
        b=0
        denomi=0
        
        for tau in taus:
            sumi=0
            totalReward=self.calcReward(tau,True)
            for i in range(len(tau[:-1,0:3]-1)):
                sa=tau[i,0:3]
                grad=np.zeros(np.shape(self.theta))
                index=self.pos[(sa[0],sa[1])]
                for act in range(len(self.MDP.moves)):
                    grad[index,act]=np.exp(self.theta[index,act])
                
                denom=np.sum(grad[index,:])

                grad[index,:]=-1*grad[index,:]/denom
                
                
                grad[index,sa[2]-1]=1+grad[index,sa[2]-1]
                

                sumi+=grad

            b+=np.square(sumi)*totalReward
            denomi+=np.square(sumi)

        x,y=np.shape(b)
        for i in range(x):
            for j in range(y):
                if b[i,j]!=0:
                    b[i,j]=b[i,j]/denomi[i,j]


        return b

                 
    def calcReward(self,tau,gradform=False):

        # calculates the reward of a trajectory
        
        
        if gradform:
            totalReward=0
            for i in range(len(tau[:,3])):
                totalReward+=(tau[i,3]*self.discount**i)
        else:
            totalReward=[]
            for i in range(len(tau[:,3])):
                totalReward.append(tau[i,3]*self.discount**i)

        return totalReward

    def viewPolicy(self):
        self.MDP.viewPolicy(self.policy)

    def viewTrajectory(self,tau,speed=0.01):
        self.MDP.viewTrajectory(tau,self.policy,speed)
    


           
R=-5     
grid=np.array([[10,0,0],[-5,R,0],[-10,-5,0],[-10,-5,0]])
initial=(3,2)
final=[(2,0),(0,0)]


MDP=GridWorld(grid,initial,final)


pol1=Policy(MDP,0.1,0.9)
a1=pol1.Reinforce(100,0.01,False,200,True,False)


pol2=pol=Policy(MDP,0.1,0.9,pol1.initial)
a2=pol2.Reinforce(100,0.01,False,200,True,True)



pol3=pol=Policy(MDP,0.1,0.9,pol1.initial)
a3=pol3.Reinforce(100,0.01,False,200,False,False)

py.plot(a1,'b')
py.plot(a2,'r')
py.plot(a3,'g')



            
    
        
        
    
                
                    
                
                
        
                    
                    
                    
                
                
            
                
        
    
