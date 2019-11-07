# Model Poisoning Attacks

This code accompanies the paper 'Analyzing Federated Learning through an Adversarial Lens' which has been accepted at ICML 2019. It assumes that the Fashion MNIST data and Census data have been downloaded to /home/data/ on the user's machine.

Dependencies: Tensorflow-1.8, keras, numpy, scipy, scikit-learn

To run federated training with 10 agents, use
```
python3 dist_train_w_attack.py --dataset=fMNIST --k=10 --C=1.0 --E=5 --T=40 --train --model_num=0
```
To run the basic targeted model poisoning attack, use
```
python3 dist_train_w_attack.py --dataset=fMNIST --k=10 --C=1.0 --E=5 --T=40 --train --model_num=0 --mal --mal_obj=single --mal_strat=converge
```

The other attacks can be found in the file `malicious_agent.py`.
