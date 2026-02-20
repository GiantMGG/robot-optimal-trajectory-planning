import csv
from importlib.resources import files

import numpy as np


class ReferenceSimulator:
    def __init__(self, config, TrajectoryPlanner):
        self.TrajectoryPlanner = TrajectoryPlanner
        self.config = config
        self.simulation_data = []
        self.X_current = []
        self.U_current = []
        self.step_size = config["sample_time"]
        self.time_current = 0

    def prepare_simulation(self, data_path):
        self.load_reference_data(data_path)
        self.parse_reference_data()
        self.maximum_simulation_length = self.compute_max_simulation_length()

    def load_reference_data(self, data_path):
        data_path = files("robot_optimal_trajectory_planning").joinpath("../../test/armJogging1.csv")
        with data_path.open("r") as file:
            reader = csv.reader(file)
            self.reference_data = [[float(val) for val in row] for row in reader]

    def parse_reference_data(self):
        # Convert the nested list to a numpy array
        arr = np.array(self.reference_data, dtype=float)
        # Transpose so columns become rows (7, 30, N)
        arr = arr.T
        # Reshape to (7, 30, -1)
        self.reference_array = arr.reshape(7, 30, -1, order="F")

    def compute_max_simulation_length(self):
        max_length = int(self.reference_array.shape[2] * 0.05 / self.config["sample_time"])
        return max_length

    def interpolate_reference(self, time_current):
        # reference data has 0.05 seconds intervals and 30*0.05 = 1.5 seconds prediction horizon
        # First find the corresponding (7,30) elements in the reference data
        # Then interpolate time_current between those elements
        # Then interpolate on this 7x30 array to get the reference for current time with shape
        # (7,config["prediction_horizon"]), by using self.config["sample_time"]
        index = int(time_current / 0.05)
        index2 = index + 1
        # interpolate between index and index2 in reference data
        arr1 = self.reference_array[:, :, index]
        arr2 = self.reference_array[:, :, index2]
        alpha = (time_current - index * 0.05) / 0.05
        arr_inter = arr1 * (1 - alpha) + arr2 * alpha

        # Interpolate along the second axis to get the reference for the current time
        reference = np.zeros((7, self.config["prediction_horizon"]))
        ref_times = np.arange(0, 30) * 0.05
        target_times = np.arange(self.config["prediction_horizon"]) * self.config["sample_time"]
        for i in range(7):
            reference[i, :] = np.interp(target_times, ref_times, arr_inter[i, :])
        return reference

    def step(self):
        self.time_current += self.step_size
        if self.time_current >= self.maximum_simulation_length:
            return False
        reference = self.interpolate_reference(self.time_current)
        self.TrajectoryPlanner.set_x0(self.X_current)
        self.TrajectoryPlanner.set_reference(reference)
        self.TrajectoryPlanner.solve()
        # Assumption: sample_time is the same as step_size
        self.X_current = self.TrajectoryPlanner.get_x_sol()[:, 1]
        self.U_current = self.TrajectoryPlanner.get_u_sol()[:, 1]


if __name__ == "__main__":
    sim = ReferenceSimulator(config={"sample_time": 0.05, "prediction_horizon": 30}, TrajectoryPlanner=None)
    sim.prepare_simulation("path/to/data")
    print(sim.interpolate_reference(10))
