from imports import *
from advertorch.attacks import LinfPGDAttack

def gen_attack ( #this attack is a PGD (Projected Gradient Descent) attack with order=L-norm inf
    model, 
    eps=0.01, #max distortion 
    nb_iter=40, #num of iters 
    eps_iter=0.002, #attack step size 
    clip_min=0.0, #min value per input dimension
    clip_max=1.0, #max value per input dimension
    targeted=False
    ):

    loss_fn = nn.CrossEntropyLoss(reduction='sum') #use sum since many attacks want the scalar loss value as a sum
    attack = LinfPGDAttack(
        model,
        loss_fn=loss_fn,
        eps=eps,
        nb_iter=40,
        eps_iter=eps_iter,
        clip_min=clip_min,
        clip_max=clip_max,
        targeted=targeted
        )
    
    return attack