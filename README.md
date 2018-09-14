# PPO Pytorch 

This repo contains an implementation of Proximal Policy Optimization. It is a minimal version of [Ilya Kostrikov's](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr) work, which in turn is depends heavily on [OpenAI baselines](https://github.com/openai/baselines) and is free of Visdom, relying rather on Tensorboard to display results. 

## In what aspects is it minimal ? 

I'm planning to power up the implementation, but, in order to have full control over all aspects, I took the liberty to remove some unecessary aspects while I'm trying to understand it all. In particular: 

* Only low-dimensional inputs (ie. no images as inputs)
* The agent doesn't have recurrent layers. 

## What to expect in a near future ? 

* Observation of more values (currently tensorboard only receives informations about rewards). I wanna add Policy loss, entropy, entropy loss, value loss... 
* A simpler implementation that I wanna make myself. 

