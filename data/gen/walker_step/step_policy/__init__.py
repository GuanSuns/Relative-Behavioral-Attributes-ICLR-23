import os


def get_step_policy():
    file_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(file_dir, 'td3_step_model.zip')
    # load models
    from stable_baselines3 import TD3
    model = TD3.load(model_path)
    return model

