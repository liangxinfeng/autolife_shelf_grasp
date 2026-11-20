import os
import unittest
from unittest.mock import patch
from autolife_shelf_grasp_306.runtime_identity import identity


class IdentityTests(unittest.TestCase):
    def test_one_namespace_for_all_channels(self):
        with patch.dict(os.environ, {'ROBOT_ID': '367', 'ROS_DOMAIN_ID': '0'}):
            robot = identity({'robot_id': 367})
            self.assertEqual(robot.topic_suffix, '0_367')
            self.assertEqual(robot.shared_key('is_navigating'), 'shared_params:0:367:is_navigating')

    def test_simulation_domain_is_respected(self):
        with patch.dict(os.environ, {'ROBOT_ID': '367', 'ROS_DOMAIN_ID': '92'}):
            self.assertEqual(identity().topic_suffix, '92_367')

    def test_deployment_can_override_old_config_without_identity_lock(self):
        with patch.dict(os.environ, {'ROBOT_ID': '367'}):
            self.assertEqual(identity({'robot_id': 345}).robot_id, '367')

    def test_invalid_identity(self):
        for values in ({'ROBOT_ID':'x'}, {'ROS_DOMAIN_ID':'233'}, {'ROS_DOMAIN_ID':'-1'}):
            with patch.dict(os.environ, values), self.assertRaises(ValueError):
                identity()
