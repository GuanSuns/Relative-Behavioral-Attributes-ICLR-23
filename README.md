# Relative Behavioral Attributes: Filling the Gap between Symbolic Goal Specification and Reward Learning from Human Preferences (ICLR 2023)

### Official python implementation of the ICLR 2023 paper: [Relative Behavioral Attributes: Filling the Gap between Symbolic Goal Specification and Reward Learning from Human Preferences](https://guansuns.github.io/pages/rba/)

**Note**: The code has been refactored for better readability. If you encounter any problem, feel free to email lguan9@asu.edu.

<p align="center">
  <img src="doc/overview.gif" alt="overview"/>
</p>


## Instructions

### Step 1. Constructing the behavior corpora

- As mentioned in the paper, ideally, the offline behavior datasets should be some publicly accessible behavior corpora like the Waymo Open Dataset. However, the primary focus of the paper is to assess the feasibility of the two proposed approaches, and therefore synthetic data was used instead.

- Also, the current implementation assumes that the behavioral dataset only demonstrates the skill to be learned. However, datasets in the real world are often not so "clean". For instance, a dataset on human driving will not exclusively demonstrate the behavior of lane changing. Furthermore, even within the lane-changing behavior, the demonstrators might perform the task under various circumstances such as different road types and traffic conditions. Hence, future research could explore how relative attributes could be acquired across various skills and domains.
  
- **Domain 1: Lane-Change**
    - To try this domain, you will need to install this customized highway environment: https://github.com/GuanSuns/Customized-Highway-Env, which is based on the original highway environment: https://github.com/Farama-Foundation/HighwayEnv.
  - To construct the training dataset, use `data/gen/highway_env/lane_change_synthetic_training.py`. This will give you a pickle file that saves all the training sample.
  - To construct the testing dataset, use `data/gen/highway_env/lane_change_synthetic_test.py`. This will give you a pickle file that saves all the testing samples.
  

- **Domain 2: Manipulator Lifting**
  - To construct the training dataset, use `data/gen/manipulator_lifting/manipulator_lifting_synthetic_training.py`. This will give you a pickle file that saves all the training sample.
  - To construct the testing dataset, use `data/gen/manipulator_lifting/manipulator_lifting_synthetic_test.py`. This will give you a pickle file that saves all the testing samples.

- **Domain 3: Walker Step**
  - To construct the training dataset, use `data/gen/walker_step/walker_step_synthetic_training.py`. This will give you a pickle file that saves all the training sample.
  - To construct the testing dataset, use `data/gen/walker_step/walker_step_synthetic_test.py`. This will give you a pickle file that saves all the testing samples.
  - Unlike the lane-change environment and the manipulator environment, where the agent's behaviors can be scripted, the step policy here has to be learned. For the purpose of experiments, we used hard-coded rewards and constraints to obtain a conditioned step policy that can produce various styles of walking behavior. The training script can be found at `environments/walker/walker_step.py`. The policy is saved in `data/gen/walker_step/step_policy/td3_step_model.zip`. We note that a functional step policy can be hard to obtain. We ran the training script multiple times, but only a few resulted in a functional policy. To load the policy, it is necessary to install Stable-Baselines3.
  


### Step 2. Learning an image-state encoder (only for Lane-Change)
- Example script: `scripts/lane_change/vae_encoder.bash`.
- In the cfg file (see the `vae_encoder.bash` file for example), you will need to set the path to the behavior dataset `dataset_dir: xxxxx`.



### Step 3. Learning an attribute reward function
- All the training scripts can be found under the `scripts` directory.
- **RBA-Global (Method 1)**
  - The first step is to learn an attribute function. For example, when language embedding is used as attribute representation, to learn the attribute function for Lane-Change, the script to use should be `scripts/lane_change/language/method_1_attr.bash`.
    - Note that in the corresponding cfg file, you need to specify the directory of the training dataset `dataset_dir` and the path to the pretrained image-state encoder `attr_func/encoder_path`.
  - The second step is to learn an attribute reward function. For example, when the language embedding is used as attribute representation, to learn an attribute function for Lane-Change, the script to use should be `scripts/lane_change/language/method_1_reward.bash`.
    - Here, in addition to the path to the training dataset and the path to the pretrained image-state encoder, you also need to specify the path to the pretrained attribute function `reward_func/attr_func_path`.
  - We also provide an example script to evaluate the performance of learned attribute functions: `runners/method_1/inspect_attr_func.py`.
  
- **RBA-Local (Method 2)**
   - In this method, we learn a reward function directly. As an example, when language embedding is used as attribute representation, to learn a reward function for Lane-Change, the script to use should be `scripts/lane_change/language/method_2_reward.bash`.
      - Note that in the corresponding cfg file, you need to specify the directory of the training dataset `dataset_dir` and the path to the pretrained image-state encoder `reward_func/encoder_path`.
  


### Step 4. Interacting with end users

- We provide example user interfaces that demonstrate the use of pretrained reward functions. The scripts are under the `human_interact` directory. In each of the script, you can set the target attribute strength in the dict `target_attrs`. You can also set the control precision by changing the value of `epsilon` in `target_attrs` (recall that we consider a trial as a successful one if the difference between the agent’s behavior and the target behavior is lower than a threshold). As discussed in the paper, the current approaches still struggle to achieve high-precision control, suggesting that further research is needed.
- For reference, the user-interface scripts load the pretrained reward functions under the `trained_models` directory.
- As mentioned in the paper, the current implementation optimizes the reward simply by sampling a large set of rollouts with the scripts or policies that we used to synthesize behavior dataset. In practice, this is similar to the case of using optimization-based planning methods or the case of using policies that are unsupervisedly learned.




  
  
  


  









