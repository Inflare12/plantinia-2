import unittest


class OfflineImportSmokeTest(unittest.TestCase):
    def test_core_modules_import(self):
        from plantinia.vision import PlantVision
        from plantinia.llm import PlantBrain
        from plantinia.robot import ArmController
        from plantinia.sensors import SensorHub

        self.assertEqual(PlantVision.confidence_state(0.2), "low")
        self.assertEqual(PlantVision.confidence_state(0.6), "review")
        self.assertEqual(PlantVision.confidence_state(0.9), "high")
        self.assertTrue(ArmController)
        self.assertTrue(SensorHub)
        self.assertTrue(PlantBrain)


if __name__ == "__main__":
    unittest.main()
