"""One robot identity for feedback, commands, services and control leases."""
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class RobotIdentity:
    robot_id: str
    domain_id: str

    @property
    def topic_suffix(self):
        return f'{self.domain_id}_{self.robot_id}'

    def shared_key(self, name):
        return f'shared_params:{self.domain_id}:{self.robot_id}:{name}'


def identity(config=None):
    config = config or {}
    configured = config.get('robot_id')
    robot = str(os.environ.get('ROBOT_ID', configured if configured is not None else ''))
    domain = str(os.environ.get('ROS_DOMAIN_ID', config.get('ros_domain_id', 0)))
    if not robot.isdecimal() or not domain.isdecimal() or not 0 <= int(domain) <= 232:
        raise ValueError('机器人编号或 ROS 域无效')
    return RobotIdentity(robot, domain)
