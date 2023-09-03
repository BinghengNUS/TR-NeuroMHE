# TR-NeuroMHE
The ***Trust-Region Neural Moving Horizon Estimation (TR-NeuroMHE)*** is an auto-tuning and adaptive optimal state estimator. It fuses a light-weight neural network with a control-theorectic MHE estimator to realize accurate estimation and fast online adaptation to environments. The neural network can be efficiently trained using the off-the-shelft trust-region method (TRM) with an extremely small amount of data. Central to our method is the Hessian trajectory of the MHE optimization problem w.r.t the tunable parameters, which is computed recursively using a Kalman filter.

|                                    Learning Piplines of the TR-NeuroMHE                                    |
:------------------------------------------------------------------------------------------------------------:
![diagram_github](https://github.com/BinghengNUS/TR-NeuroMHE/assets/70559054/948f465e-9a73-42e1-8e66-a3c8d9c91904)

## Table of contents
1. [Dependency Packages](#Dependency-Packages)
2. [How to Use](#How-to-Use)
      1. [Training and Evaluation](#Training-and-Evaluation)
      2. [RMSE Reproduction](#RMSE-Reproduction)
      3. [Applications to other robots](#Applications-to-other-robots)
3. [Contact Us](#Contact-Us)

## 1. Dependency Packages
Please make sure that the following packages have already been installed before running the source code.
* CasADi: version 3.5.5 Info: https://web.casadi.org/
* Numpy: version 1.23.0 Info: https://numpy.org/
* Pytorch: version 1.12.0+cu116 Info: https://pytorch.org/
* Matplotlib: version 3.3.0 Info: https://matplotlib.org/
* Python: version 3.9.12 Info: https://www.python.org/
* Scipy: version 1.8.1 Info: https://scipy.org/
* Pandas: version 1.4.2 Info: https://pandas.pydata.org/
* scikit-learn: version 1.0.2 Info: https://scikit-learn.org/stable/whats_new/v1.0.html

## 2. How to Use
The training process for TR-NeuroMHE is both efficient and straightforward to setup. To reproduce the simulation results presented in the paper, simply follow the steps outlined below, sequentially, after downloading and decompressing all the necessary folders.

### Training and Evaluation
1. Download '**processed_data.zip**' and '**predictions.tat.xz**' from https://download.ifi.uzh.ch/rpg/NeuroBEM/. The former file is utilized for training TR-NeuroMHE, whereas the latter serves the purpose of evaluation and comparison with TR-NeuroBEM.
2. Relocate the folder '**bem+nn**' from the decomprassed archive '**predictions.tat.xz**' to the downloaded folder '**SecVII-A (source code)**', and place the decompressed '**processed_data.zip**' within the folder '**Source_code_TR_NeuroMHE**' as well.
3. Run the Python file '**main_code_trust_region_neuromhe.py**'.
4. In the prompted terminal interface, you will be asked to select whether to train or evaluate TR-NeuroMHE.
   * Training: type 'train' without the quotation mark in the terminal.
   * Evaluation: type 'evaluate' without the quotation mark in the terminal. We evaluate and compare the estimation performance of TR-NeuroMHE with a state-of-the-art estimator, NeuroBEM [[1]](#1), on its complete test dataset. The testset includes 13 agile flight trajectories which were unseen in training. The following table summarizes the parameters of these trajectories. Note that you can skip the training process and directly evaluate the performance using the trained neural network model '**nn_para_TR_NeuroMHE_1.npy**' to reproduce the RMSE results presented in our paper. The retained model is saved in the folder '**trained_data**'.
   
|                                         Trajectory Parameters of NeuroBEM Test Dataset                                           |
:----------------------------------------------------------------------------------------------------------------------------------:
![test dataset](https://github.com/RCL-NUS/NeuroMHE/assets/70559054/afbdc415-288b-4938-8bc9-7b18c59d6f40)

One advantage of NeuroBEM is that its accuracy only declines by 20% when the training dataset encompasses a limited portion of the velocity-range space compared to the test dataset. To assess the performance of our TR-NeuroMHE in this scenario, we select an extremely short flight segment (0.25 s) from an agile figure-8 trajectory, covering a limited velocity range of 8 m/s to 12 m/s. The following figures present a comparison of the velocity-range space in the world frame between the training sets and the partial test sets.
        Velocity-Range Space: Training Sets        |      Velocity-Range Space: Partial Test Sets
:---------------------------------------------------------------:|:--------------------------------------------------------------:
![3d_velocityspace_training_very_slow](https://github.com/BinghengNUS/TR-NeuroMHE/assets/70559054/eb1fbea7-e0da-4f13-b689-156bdd721c8b) | ![3d_velocityspace_test](https://github.com/BinghengNUS/TR-NeuroMHE/assets/70559054/96b4075b-8b4d-49a4-a432-2e5d5e8050d6)







## References
<a id="1">[1]</a> 
L. Bauersfeld, E. Kaufmann, P. Foehn, S. Sun, and D. Scaramuzza, "NeuroBEM: Hybrid Aerodynamic Quadrotor Model", ROBOTICS: SCIENCE AND SYSTEM XVII,2021.
