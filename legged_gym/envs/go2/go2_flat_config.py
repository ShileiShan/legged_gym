from legged_gym.envs import Go2RoughCfg, Go2RoughCfgPPO

class Go2FlatCfg( Go2RoughCfg ):
    class env( Go2RoughCfg.env ):
        num_observations = 48
  
    class terrain( Go2RoughCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False

    # class init_state( Go2RoughCfg.init_state ):
    #     pos = [0.0, 0.0, 0.6] # x,y,z [m]
    #     default_joint_angles = { # = target angles [rad] when action = 0.0
    #         "LF_HAA": 0.0,
    #         "LH_HAA": 0.0,
    #         "RF_HAA": -0.0,
    #         "RH_HAA": -0.0,

    #         "LF_HFE": 0.4,
    #         "LH_HFE": -0.4,
    #         "RF_HFE": 0.4,
    #         "RH_HFE": -0.4,

    #         "LF_KFE": -0.8,
    #         "LH_KFE": 0.8,
    #         "RF_KFE": -0.8,
    #         "RH_KFE": 0.8,
    #     }

    class rewards( Go2RoughCfg.rewards ):
        max_contact_force = 350.
        class scales ( Go2RoughCfg.rewards.scales ):
            orientation = -5.0
            torques = -0.000025
            feet_air_time = 2.
            # feet_contact_forces = -0.01
    class asset( Go2RoughCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf'
        name = "go1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = True

    class asset( Go2RoughCfg.asset ):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/anymal_c/urdf/anymal_c.urdf"
        name = "anymal_c"
        foot_name = "FOOT"
        penalize_contacts_on = ["SHANK", "THIGH"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class commands( Go2RoughCfg.commands ):
        heading_command = False
        resampling_time = 4.
        class ranges( Go2RoughCfg.commands.ranges ):
            lin_vel_x = [0.5, 1.0] # min max [m/s]  
            lin_vel_y = [-0.8, 0.8]   # min max [m/s]
            ang_vel_yaw = [0, 0]   # min max [rad/s]
            heading = [-3.14, 3.14]

    class domain_rand( Go2RoughCfg.domain_rand ):
        friction_range = [0., 1.5] # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.

class Go2FlatCfgPPO( Go2RoughCfgPPO ):
    class policy( Go2RoughCfgPPO.policy ):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm( Go2RoughCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner ( Go2RoughCfgPPO.runner):
        run_name = ''
        experiment_name = 'flat_go2'
        load_run = -1
        max_iterations = 300
