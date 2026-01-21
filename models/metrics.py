import torch
from torchmetrics import Metric
from .losses import *

class LossMetric(Metric):
    def __init__(self, loss_func, loss_kwargs={}, **kwargs):
        super().__init__()
        self.loss_func = loss_func
        self.loss_kwargs = loss_kwargs
        self.add_state("loss", default=torch.tensor(0.), dist_reduce_fx="mean")
        self.add_state("step", default=torch.tensor(0.), dist_reduce_fx="mean")

    def update(self, preds, targets):
        self.loss += self.loss_func(preds, targets, **self.loss_kwargs)
        self.step += 1.0
        
    def compute(self):
        return self.loss / self.step
    
    def __repr__(self):
        return '%e ' % self.compute()

class MeanL21LossInvariant(LossMetric):
    def __init__(self, **kwargs):
        super().__init__(loss_func=l21_loss_affine_invariant, loss_kwargs={'masked':True, 'eps':0})

class MeanL22LossInvariant(LossMetric):
    def __init__(self, **kwargs):
        print("L22 LOSS mean")
        super().__init__(loss_func=l22_loss_affine_invariant, loss_kwargs={'masked':True, 'eps':0})

class MeanL2Loss(LossMetric):
    def __init__(self, **kwargs):
        super().__init__(loss_func=l2_loss, loss_kwargs={'masked':True, 'eps':0})

class TRELoss(LossMetric):
    def __init__(self, **kwargs):
        super().__init__(loss_func=TRE_loss_grid, loss_kwargs={'masked':True, 'eps':0})

class TRELossSoftMax(LossMetric):
    def __init__(self, **kwargs):
        super().__init__(loss_func=TRE_loss_soft_max, loss_kwargs={'masked':True, 'eps':0})

class MLP_Cross(LossMetric):
    def __init__(self, **kwargs):
        super().__init__(loss_func=classification_onehot_loss)

class MLP_CrossMulti(LossMetric):
    def __init__(self, **kwargs):
        super().__init__(loss_func=classification_multihot_loss)

class MLP_CrossMulti2(LossMetric):
    def __init__(self, **kwargs):
        super().__init__(loss_func=classification_multihot_loss2)


class MLP_CrossMultiOrder(LossMetric):
    def __init__(self, **kwargs):
        super().__init__(loss_func=classification_multihot_loss_stack_order)


class MLP_CrossVec(LossMetric):
    def __init__(self, **kwargs):
        super().__init__(loss_func=classification_cross_multihot_vec)

class MLP_CorrectStack(LossMetric):
    def __init__(self, **kwargs):
        super().__init__(loss_func=classification_correct_stack)

class MLP_CorrectRot(LossMetric):
    def __init__(self, **kwargs):
        super().__init__(loss_func=classification_correct_rot)

class MLP_CorrectOrder(LossMetric):
    def __init__(self, **kwargs):
        super().__init__(loss_func=classification_correct_order)

class MLP_CorrectVec(LossMetric):
    def __init__(self, **kwargs):
        super().__init__(loss_func=classification_correct_vec)