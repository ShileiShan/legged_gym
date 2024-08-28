from legged_gym.envs import Go1RoughCfg, Go1RoughCfgPPO

class Go1FlatCfg( Go1RoughCfg ):
    class env( Go1RoughCfg.env ):
        num_observations = 48
  
    class terrain( Go1RoughCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False

    class rewards( Go1RoughCfg.rewards ):
        max_contact_force = 350.
        class scales ( Go1RoughCfg.rewards.scales ):
            orientation = -5.0
            torques = -0.000025
            feet_air_time = 2.
            # feet_contact_forces = -0.01

    class commands( Go1RoughCfg.commands ):
        heading_command = False
        resampling_time = 4.
        class ranges( Go1RoughCfg.commands.ranges ):
            lin_vel_x = [0.5, 1.0] # min max [m/s]  
            lin_vel_y = [0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [0, 0]   # min max [rad/s]
            heading = [0, 0]

    class domain_rand( Go1RoughCfg.domain_rand ):
        friction_range = [0.5, 1.5] # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.

class Go1FlatCfgPPO( Go1RoughCfgPPO ):
    class policy( Go1RoughCfgPPO.policy ):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm( Go1RoughCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner ( Go1RoughCfgPPO.runner):
        run_name = ''
        experiment_name = 'flat_go1'
        load_run = -1
        max_iterations = 300
