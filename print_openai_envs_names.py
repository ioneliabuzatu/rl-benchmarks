import sys
from gym import envs

def print_envs():
    """
    it prints the openai env names, if you pass a partial env name 
    then it will print only the env containing the partial env name
    """

    all_envs = envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]

    try:
        user_env_name = sys.argv[1]
        if user_env_name is not None:
            print(*[name for name in env_ids if user_env_name in name], sep="\n")

    except IndexError:        
        print("\n".join(env_ids))


if __name__ == "__main__":
    print_envs()
