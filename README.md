# TR-NeuroMHE
The ***Trust-Region Neural Moving Horizon Estimation (TR-NeuroMHE)*** is an adaptive optimal state estimator tuned using the trust-region method. Accurate disturbance estimation is essential for safe robot operations. The recently proposed NeuroMHE, which uses a portable neural network to model the MHE's weightings, has shown promise in further pushing the accuracy and efficiency boundary. Currently, NeuroMHE is trained through gradient descent, with its gradient computed recursively using a Kalman filter. In this work, we show that much of computation already used to obtain the gradient, especially the Kalman filter, can be efficiently reused to compute the MHE Hessian for training TR-NeuroMHE.

|                                    Learning Piplines of The Proposed TR-NeuroMHE                                    |
:------------------------------------------------------------------------------------------------------------:
![plot_in_github](https://github.com/BinghengNUS/TR-NeuroMHE/assets/70559054/84707962-c0f2-40ac-ae6a-63287116c80c)

Please find out more details in 
   * our paper https://arxiv.org/abs/2309.05955
   * NeuroMHE trained using gradient descent https://github.com/RCL-NUS/NeuroMHE


## Table of contents
1. [Dependency Packages](#Dependency-Packages)
2. [How to Use](#How-to-Use)
      1. [Training and Evaluation](#Training-and-Evaluation)
      2. [RMSE Reproduction](#RMSE-Reproduction)
      3. [Applications to other robots](#Applications-to-other-robots)
3. [Acknowledgement](#Acknowledgement)
4. [Contact Us](#Contact-Us)

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

The comparative results in terms of RMSE are summarized in the following table.
|                               Estimation Errors (RMSES) Comparisons on the NeuroBEM Test Dataset                                 |
:----------------------------------------------------------------------------------------------------------------------------------:
![RMSE_Comparison_NeuroBEM_test_dataset](https://github.com/BinghengNUS/TR-NeuroMHE/assets/70559054/abce500e-dacf-4254-a039-fc9504c09ad4)




### RMSE Reproduction
1. In the folder '**Check_RMSE**', run the MATLAB file '**RMSE_vector_error_TR_NeuroMHE.m**' to replicate the RMSE results presented in the above table. The results are obtained through vector error (i.e., $e_{f}=\sqrt{(d_{f_x}-\hat d_{f_x})^2 + (d_{f_y}-\hat d_{f_y})^2 + (d_{f_z}-\hat d_{f_z})^2}$) with the force vector expressed in the body frame. You can also run the corresponding Python files for the RMSE reproduction. These files have the same names as the MATLAB counterparts but end with '**.py**'.

   Note that the residual force data provided in the NeuroBEM dataset (columns 36-38 in the file 'predictions.tar.xz') was computed using the initially reported mass of 0.752 kg [[1]](#1) instead of the later revised value of 0.772 kg (See the NeuroBEM's website). As a result, we refrain from utilizing this data to compute NeuroBEM's RMSE. The rest of the dataset remains unaffected, as the residual force data is provided purely for users' convenience. It can be calculated from the other provided data including the mass value, as explained on https://rpg.ifi.uzh.ch/neuro_bem/Readme.html.

2. In the subfolder '**MATLAB_code_for_mass_verification**', run the MATLAB file '**residual_force_XXX.m**' to demonstrate the mass verification, where '**XXX**' represents the name of the test trajectory, such as '**3D_Circle_1**'.


### Applications to other robots
Please note that although we demonstrated the effectiveness of our approach using a quadrotor, the proposed method is general and can be applied to robust adaptive control for other robotic systems. Only minor modifications in our code are needed for such applications. To illustrate, we can take the source code in the folder '**Source_code_TR-NeuroMHE**' as an example and proceed as follows:
   * Update the robotic dynamics model in the Python file '**Uav_Env.py**';
   * Add a robotic controller in the Python file '**Uav_mhe_SL_Hessian_trust_region_neural.py**';
   * Update the simulation environment for training and evaluation in the Python file '**main_code_trust_region_neuromhe.py**'.

## 3. Acknowledgement
We thank Leonard Bauersfeld for the help in using the flight dataset of NeuroBEM.

## 4. Contact Us
If you encounter a bug in your implementation of the code, please do not hesitate to inform me.
* Name: Mr. Bingheng Wang
* Email: wangbingheng@u.nus.edu

## References
<a id="1">[1]</a> 
L. Bauersfeld, E. Kaufmann, P. Foehn, S. Sun, and D. Scaramuzza, "NeuroBEM: Hybrid Aerodynamic Quadrotor Model", ROBOTICS: SCIENCE AND SYSTEM XVII,2021.
