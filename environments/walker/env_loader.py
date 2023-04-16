from environments.walker import walker_base


def load_walker(task_name, task_kwargs=None, environment_kwargs=None,
                visualize_reward=False):
    return build_environment(task_name, task_kwargs,
                             environment_kwargs, visualize_reward)


def build_environment(task_name, task_kwargs=None,
                      environment_kwargs=None, visualize_reward=False):
    domain = walker_base
    task_kwargs = task_kwargs or {}
    if environment_kwargs is not None:
        task_kwargs = dict(task_kwargs, environment_kwargs=environment_kwargs)
    env = domain.SUITE[task_name](**task_kwargs)
    env.task.visualize_reward = visualize_reward
    return env
