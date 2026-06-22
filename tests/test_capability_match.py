import sys
import types
import unittest

import numpy as np


# The environment imports plotting modules even though these tests do not plot.
if 'matplotlib' not in sys.modules:
    matplotlib = types.ModuleType('matplotlib')
    matplotlib.pyplot = types.ModuleType('matplotlib.pyplot')
    matplotlib.patches = types.ModuleType('matplotlib.patches')
    matplotlib.animation = types.ModuleType('matplotlib.animation')
    matplotlib.offsetbox = types.ModuleType('matplotlib.offsetbox')
    matplotlib.animation.FuncAnimation = object
    matplotlib.offsetbox.OffsetImage = object
    matplotlib.offsetbox.AnnotationBbox = object
    sys.modules['matplotlib'] = matplotlib
    sys.modules['matplotlib.pyplot'] = matplotlib.pyplot
    sys.modules['matplotlib.patches'] = matplotlib.patches
    sys.modules['matplotlib.animation'] = matplotlib.animation
    sys.modules['matplotlib.offsetbox'] = matplotlib.offsetbox

from env.task_env import TaskEnv


class CapabilityMatchTest(unittest.TestCase):
    def test_binary_match_reuses_original_expression(self):
        cases = [
            ([1, 0, 1], [0, 2, 1], 1.0),
            ([1, 0, 0], [0, 2, 1], 0.0),
            ([1, 1, 0], [-1, 0, 2], 0.0),
        ]
        for abilities, status, expected in cases:
            with self.subTest(abilities=abilities, status=status):
                actual = TaskEnv.compute_capability_match_from_existing_logic(
                    np.asarray(abilities), np.asarray(status))
                self.assertEqual(expected, actual)

    def test_mask_matches_pre_refactor_logic(self):
        env = TaskEnv((1, 1), (2, 2), (4, 4), traits_dim=3, seed=7)
        agent_id = 0
        agent = env.agent_dic[agent_id]

        expected = np.ones(env.tasks_num, dtype=bool)
        for task in env.task_dic.values():
            if not task['feasible_assignment']:
                ability = np.maximum(np.minimum(task['status'], agent['abilities']), 0.)
                if ability.sum() > 0:
                    expected[task['ID']] = False

        np.testing.assert_array_equal(expected, env.get_contributable_task_mask(agent_id))


if __name__ == '__main__':
    unittest.main()
