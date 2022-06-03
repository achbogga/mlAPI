import json
import os

from get_sample_request_json import get_json_data, unit_test_gen
from locust import HttpUser, LoadTestShape, between, task


class ProfileLoad(LoadTestShape):
    """
    This load profile starts at 0 and steps up by step_users
    increments every tick, up to target_users.  After reaching
    target_user level, load will stay at target_user level
    until time_limit is reached.
    """

    target_users = 1
    step_users = 1  # ramp users each step
    time_limit = 3600  # seconds

    def tick(self):
        num_steps = self.target_users / self.step_users
        run_time = round(self.get_run_time())

        if run_time < self.time_limit:
            if num_steps < run_time:
                user_count = num_steps * self.step_users
            else:
                user_count = self.target_users
            return (user_count, self.step_users)
        else:
            return None


class TritonUser(HttpUser):
    wait_time = between(0.2, 0.2)

    @task()
    def bert(self):
        response = self.client.post(self.url1, data=json.dumps(self.data))

    def on_start(self):
        # with open('sample_request.json') as f:
        #     self.data = json.load(f)
        no_of_images = 1
        INGRESS_HOST = os.environ["INGRESS_HOST"]
        INGRESS_PORT = os.environ["INGRESS_PORT"]
        unit_test_gen_obj = unit_test_gen(no_of_images)
        json_data = {}
        inputs_1 = {
            "name": "input.1",
            "shape": [64, 3, 256, 256],
            "datatype": "FP32",
            "parameters": {"binary_data": False},
        }
        outputs_1 = {
            "name": "1510",
            "shape": [4],
            "datatype": "FP32",
            "parameters": {"binary_data": True},
        }
        request_wise_batched_data = get_json_data(
            "convnext_onnx",
            unit_test_gen_obj,
            no_of_images=no_of_images,
            batch_size=no_of_images,
            server_url=str(INGRESS_HOST) + ":" + str(INGRESS_PORT),
        )
        inputs_1["data"] = request_wise_batched_data[0].flatten().tolist()
        print(inputs_1["data"])
        json_data["inputs"] = [inputs_1]
        json_data["outputs"] = [outputs_1]
        self.data = json_data

        self.url1 = "{}/v2/models/{}/infer".format(
            self.environment.host, "convnext_onnx"
        )
