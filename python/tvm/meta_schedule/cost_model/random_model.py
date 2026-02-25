# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Random cost model
"""
from typing import List, Optional, Tuple, Union

from ..cost_model import PyCostModel
from ..runner import RunnerResult
from ..search_strategy import MeasureCandidate
from ..tune_context import TuneContext
from ..utils import derived_object  # type: ignore

# kyunam
from ...runtime import NDArray
from ..feature_extractor import FeatureExtractor
from datetime import datetime
import numpy as np
import os
import json
from math import floor as math_floor


@derived_object
class RandomModel(PyCostModel):
    """Random cost model

    Parameters
    ----------
    random_state : Union[Tuple[str, np.ndarray, int, int, float], dict]
        The random state of the random number generator.
    path : Optional[str]
        The path of the random cost model.
    max_range : Optional[int]
        The maximum range of random results, [0, max_range].

    Reference
    ---------
    https://numpy.org/doc/stable/reference/random/generated/numpy.random.get_state.html
    """

    random_state: Union[Tuple[str, np.ndarray, int, int, float], dict]
    path: Optional[str]

    def __init__(
        self,
        *,
        seed: Optional[int] = None,
        path: Optional[str] = None,
        max_range: Optional[int] = 100,
    ):

        super().__init__()
        if path is not None:
            self.load(path)
        else:
            np.random.seed(seed)
            self.random_state = np.random.get_state()
        self.max_range = max_range

        # kyunam
        # Init feature extractor
        self.extractor = FeatureExtractor.create("per-store-feature")

    def load(self, path: str) -> None:
        """Load the cost model from given file location.

        Parameters
        ----------
        path : str
            The file path.
        """
        import numpy as np  # type: ignore # pylint: disable=import-outside-toplevel

        self.random_state = tuple(np.load(path, allow_pickle=True))  # type: ignore

    def save(self, path: str) -> None:
        """Save the cost model to given file location.

        Parameters
        ----------
        path : str
            The file path.
        """
        import numpy as np  # type: ignore # pylint: disable=import-outside-toplevel

        np.save(path, np.array(self.random_state, dtype=object), allow_pickle=True)

    def update(
        self,
        context: TuneContext,
        candidates: List[MeasureCandidate],
        results: List[RunnerResult],
    ) -> None:
        """Update the cost model given running results.

        Parameters
        ----------
        context : TuneContext,
            The tuning context.
        candidates : List[MeasureCandidate]
            The measure candidates.
        results : List[RunnerResult]
            The running results of the measure candidates.
        """

        # kyunam
        # Use RandomModel so that we can collect diverse (features, power_value) data
        # If we were to use XGBModel, which selects the candidates with the highest cost model output,
        # most of the power data would be in a narrow range
        # RandomModel just randomly selects candidates so the power data are expected to be in a wider range
        # print(f"RandomModel.update() is called with {len(candidates)} MeasureCandidates and {len(results)} RunnerResults")  # kyunam

        # Do nothing, but just record the data
        def _feature(x: NDArray) -> np.ndarray:
            return x.numpy().astype("float32")

        def _mean_cost(x: RunnerResult) -> float:
            if not x.run_secs:
                return 1e10
            return float(np.median([float(s) for s in x.run_secs]))

        new_features = [_feature(x) for x in self.extractor.extract_from(context, candidates)]
        new_mean_costs = [_mean_cost(x) for x in results]

        # Recover power and latency data from final_value from C++ backend
        powers = []
        delays = []
        for cost in new_mean_costs:
            recovered_power = math_floor((cost / 1e3) * 10.0) / 10.0
            recovered_delay = cost - (recovered_power * 1e3)
            if recovered_power > 1e6 or recovered_delay == 0:
                # Account for failed candidates
                recovered_delay = 1e10
            delays.append(recovered_delay)
            powers.append(recovered_power)
        new_mean_costs = powers

        # Power model training pipeline expects the npz files in the unit MW
        for i in range(len(new_mean_costs)):
            new_mean_costs[i] = new_mean_costs[i] / 1e6

        # Filter instances with no features
        new_mean_costs = [c for i, c in enumerate(new_mean_costs) if len(new_features[i]) != 0]
        new_mean_costs_np = np.array(new_mean_costs).astype("float32")
        new_features = [f for f in new_features if len(f) != 0]
        if not new_features:
            return

        # Save IRs, schedules, per-store feature values, and power consumption values to a file
        eyas_dir = os.path.dirname(os.path.abspath(__file__)) + "/../../../../eyas/"
        os.makedirs(f"{eyas_dir}/dataset", exist_ok=True)
        formatted_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        features_to_save = np.array(new_features, dtype=object)  # np array of N 2D np arrays
        costs_to_save = new_mean_costs_np  # np array of N floats
        irs_to_save = [str(cand.sch.mod) for cand in candidates]  # list of N strings
        schedules_to_save = [str(cand.sch.trace) for cand in candidates]  # list of N strings
        print(f"Saving {costs_to_save.shape[0]} hardware data: ")
        print(costs_to_save)
        np.savez(f"{eyas_dir}/dataset/{formatted_time}.npz", features=features_to_save, costs=costs_to_save)
        with open(f"{eyas_dir}/dataset/{formatted_time}_ir.json", "w", encoding="utf-8") as f:
            json.dump(irs_to_save, f, ensure_ascii=False, indent=2)
        with open(f"{eyas_dir}/dataset/{formatted_time}_schedule.json", "w", encoding="utf-8") as f:
            json.dump(schedules_to_save, f, ensure_ascii=False, indent=2)


    def predict(
        self, context: TuneContext, candidates: List[MeasureCandidate]
    ) -> np.ndarray:  # type: ignore # pylint: disable=used-before-assignment
        """Update the cost model given running results.

        Parameters
        ----------
        context : TuneContext,
            The tuning context.
        candidates : List[MeasureCandidate]
            The measure candidates.

        Return
        ------
        result : np.ndarray
            The predicted running results.
        """
        import numpy as np  # type: ignore # pylint: disable=import-outside-toplevel

        np.random.set_state(self.random_state)
        # TODO(@zxybazh): Use numpy's RandState object:
        # https://numpy.org/doc/1.16/reference/generated/numpy.random.RandomState.html#numpy.random.RandomState
        result = np.random.rand(len(candidates)) * self.max_range  # type: ignore
        self.random_state = np.random.get_state()
        return result
