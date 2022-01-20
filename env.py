import re
import numpy as np
from sklearn.model_selection import StratifiedKFold
from preprocess import getDL
from const import *
from random import randint


class env():
    def initiate(self):
        self.env=getDL()
        self.indics=len(self.env)
        self.num_image=len(list(self,env))
        self.sampled=np.ones(IMAGE_SIZE)
        self.mask=np.array([1 for i in range(NUM_ACTION)])
        self.buffer=[]
        self.count=0
        self.current_image
        self.ground_truth
        self.location
        self.reset_image()
        self.check_mask()
        state=self.get_state()

        return state
        
    def step(self,action):
        move=(ACTION_X[action],ACTION_Y[action])
        if(move[0]==0 and move[1]==0):
            self.reset_image()
            self.mark_as_sampled()
            self.check_mask()
            self.buffer.append(self.current_image[:][self.location])
            self.count+=1

        else:
            self.location[0]+=move[0]
            self.location[1]+=move[1]
            self.mark_as_sampled()
            self.check_mask()
            self.buffer.append(self.current_image[:][self.location])
            self.count+=1

        done=False
        if(self.count==SAMPLE_SIZE):
            done=True
        state=self.get_state()
        reward=self.get_reward()
        return state,reward,done

    
    def get_reward(self):
        #FIXME
        return 0

    
    def get_state(self):
        return [self.current_image[0],self.current_image[1],self.current_image[2],self.sampled]

    def reset_image(self):
        #random an initial image
        index=randint(0,self.num_image-1)
        self.current_image=self.env[index][0]
        self.ground_truth=self.env[index][1]

        #reset location
        x=randint(0,(IMAGE_SIZE[0]-PATCH_SIZE[0])/STEP)*STEP
        y=randint(0,(IMAGE_SIZE[0]-PATCH_SIZE[0])/STEP)*STEP
        self.location=(x,y)
    
    def save_sample(self):
        self.buffer.append(self.current_image[:][self.location[0]:self.location[0]+PATCH_SIZE[0]][self.location[1]:self.location[1]+PATCH_SIZE[1]])


    def mark_as_sampled(self):
        self.sampled[self.location[0]:self.location[0]+PATCH_SIZE[0]][self.location[1]:self.location[1]+PATCH_SIZE[1]]=0
    
    def check_boundary(self,x,y):
        if(x<IMAGE_SIZE[0] and x>=0 and y<IMAGE_SIZE[1] and y>=0):
            return True
        else:
            return False

    def check_mask(self):
        for i in range(NUM_ACTION-1):
            temp_x=ACTION_X[i]+self.location[0]
            temp_y=ACTION_Y[i]+self.location[1]
            if(self.check_boundary(temp_x,temp_y)):
                self.mask[i]=1
            else:
                self.mask[i]=0
    
        