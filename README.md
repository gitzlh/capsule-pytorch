# capsule-pytorch
an extremyly each-to-follow pytorch implement of Hinton's capsule network. 


## Motivation
It is hard to believe that if you search "capsule+pytorch" in github, the top three repos all contain serious mistakes (see their [corresponding](https://github.com/gram-ai/capsule-networks/issues) [issues](https://github.com/timomernick/pytorch-capsule) [pages](https://github.com/higgsfield/Capsule-Network-Tutorial)). 

The mistakes include:

1. wrong softmax dimention for the calculation of c_ij in Eq.3 of the original paper.
2. wrong margin loss function.
3. wrong dimention to which the squash function is applied.

Once these mistakes are corrected, their models cannot learn at all. Maybe there are some other mistakes. 

Besides, all these repos are based on pytorch 0.3 and you need to treak the code.

PS: I did find a bug-free repos from [here](https://github.com/manuelsh/capsule-networks-pytorch).


## The best tutorials of capsule network (which, of course, is not the [original paper](https://arxiv.org/abs/1710.09829)） 

[Understanding Hinton’s Capsule Networks](https://pechyonkin.me/capsules-1/)

## run
```
python3 train.py
```
