
from lerobot.robots import make_robot_from_config
from lerobot.scripts.lerobot_record import RecordConfig
from lerobot.configs.parser import draccus
import torch

def test():
    cli_args = [
        "--robot.type", "so101_follower",
        "--robot.cameras", '{"webcam": {"type": "opencv", "index_or_path": 0}}'
    ]
    cfg = draccus.parse(config_class=RecordConfig, args=cli_args)
    print(f"Config cameras: {cfg.robot.cameras}")
    
    robot = make_robot_from_config(cfg.robot)
    print(f"Robot cameras: {robot.cameras}")
    
    # We don't connect to arm, but we can check the cameras dict
    obs = robot.get_observation()
    print(f"Observation keys: {list(obs.keys())}")

if __name__ == "__main__":
    test()
