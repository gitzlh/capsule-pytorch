# capsule-pytorch
an extremyly each-to-follow pytorch implement of Hinton's capsule network. 


## Motivation
It is hard to believe that if you search "capsule+pytorch" in github, the top three repos all contain serious mistakes (see their [corresponding](https://github.com/gram-ai/capsule-networks/issues) [issues](https://github.com/timomernick/pytorch-capsule) [pages](https://github.com/higgsfield/Capsule-Network-Tutorial)). 

The mistakes include:

1. wrong softmax dimention for the calculation of c_ij in Eq.3 of the original paper.
2. wrong margin loss function.
3. wrong dimention to which the squash function is applied.

Once these mistakes are corrected, their models cannot learn at all. I guess there are some other mistakes. 

Besides, most of the  pytorch-implemented capsule network are based on pytorch 0.3 and are a bit of outdated.




## run
```
python3 train.py
```

## Thanks:

- The best tutorials of capsule network (which, of course, is not the [original paper](https://arxiv.org/abs/1710.09829)） 
[Understanding Hinton’s Capsule Networks](https://pechyonkin.me/capsules-1/)
- The only bug-free pytorch implementation: [capsule-netwprks-pytorch](https://github.com/manuelsh/capsule-networks-pytorch).

- A well-structured implemention: [pytorch-capsule](https://github.com/timomernick/pytorch-capsule).

