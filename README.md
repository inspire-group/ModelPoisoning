# Model Poisoning Attacks

This code accompanies the paper 'Analyzing Federated Learning through an Adversarial Lens' which has been accepted at ICML 2019. It assumes that the Fashion MNIST data and Census data have been downloaded to /home/data/ on the user's machine.

Dependencies: Tensorflow-1.8, keras, numpy, scipy, scikit-learn

To run federated training with 10 agents and standard averaging based aggregation, use
```
python dist_train_w_attack.py --dataset=fMNIST --k=10 --C=1.0 --E=5 --T=40 --train --model_num=0 --gar=avg
```
To run the basic targeted model poisoning attack, use
```
python dist_train_w_attack.py --dataset=fMNIST --k=10 --C=1.0 --E=5 --T=40 --train --model_num=0 --mal --mal_obj=single --mal_strat=converge --gar=avg
```

To run the alternating minimization attack with distance constraints with the parameters used in the paper, run
```
python dist_train_w_attack.py --dataset=fMNIST --k=10 --C=1.0 --E=5 --T=40 --train --model_num=0 --mal --mal_obj=single --mal_strat=converge_train_alternate_wt_o_dist_self --rho=1e-4 --gar=avg --ls=10 --mal_E=10
```

The function of the various parameters that are set by `utils/globals_vars.py` is given below.

| Parameter   | Function                                               |
|-------------|--------------------------------------------------------|
| --gar       | Gradient Aggregation Rule                              |
| --eta       | Learning Rate                                          |
| --k         | Number of agents                                       |
| --C         | Fraction of agents chosen per time step                |
| --E         | Number of epochs for each agent                        |
| --T         | Total number of iterations                             |
| --B         | Batch size at each agent                               |
| --mal_obj   | Single or multiple targets                             |
| --mal_num   | Number of targets                                      |
| --mal_strat | Strategy to follow                                     |
| --mal_boost | Boosting factor                                        |
| --mal_E     | Number of epochs for malicious agent                   |
| --ls        | Ratio of benign to malicious steps in alt. min. attack |
| --rho       | Weighting factor for distance constraint               |

The other attacks can be found in the file `malicious_agent.py`.
