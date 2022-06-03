#!/usr/bin/env python
"""
Nvidia Triton Image Client
A client function to query triton inference server for classification
The only requirement is to install with the following pip command:
    pip3 install tritonclient[all]
created on May, 2022
@author : aboggaram@iunu.com
"""
import string
import sys
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from attrdict import AttrDict
from PIL import Image
from tritonclient.utils import InferenceServerException, triton_to_np_dtype

if sys.version_info >= (3, 0):
    import queue
else:
    import Queue as queue


def get_slice_generator(np_image, slice_width, slice_height):
    """
    A generator function to generate slices given a numpy image
    Parameters
    ----------
    np_image : np.ndarray
    slice_width : int
    slice_height : int

    Returns
    ----------
    generator_obj
    """
    for x in range(0, np_image.shape[0], slice_width):
        for y in range(0, np_image.shape[1], slice_height):
            np_slice = np_image[x : x + slice_width, y : y + slice_height]
            yield Image.fromarray(np_slice)


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


# Callback function used for async_stream_infer()
def completion_callback(user_data, result, error):
    # passing error raise and handling out
    user_data._completed_requests.put((result, error))


def parse_model(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    if len(model_metadata.inputs) != 1:
        raise Exception("expecting 1 input, got {}".format(len(model_metadata.inputs)))
    if len(model_metadata.outputs) != 1:
        raise Exception(
            "expecting 1 output, got {}".format(len(model_metadata.outputs))
        )

    if len(model_config.input) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config.input)
            )
        )

    input_metadata = model_metadata.inputs[0]
    input_config = model_config.input[0]
    output_metadata = model_metadata.outputs[0]

    if output_metadata.datatype != "FP32":
        raise Exception(
            "expecting output datatype to be FP32, model '"
            + model_metadata.name
            + "' output type is "
            + output_metadata.datatype
        )

    # Output is expected to be a vector. But allow any number of
    # dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
    # }, { 10, 1, 1 } are all ok). Ignore the batch dimension if there
    # is one.
    output_batch_dim = model_config.max_batch_size > 0
    non_one_cnt = 0
    for dim in output_metadata.shape:
        if output_batch_dim:
            output_batch_dim = False
        elif dim > 1:
            non_one_cnt += 1
            if non_one_cnt > 1:
                raise Exception("expecting model output to be a vector")

    # Model input must have 3 dims, either CHW or HWC (not counting
    # the batch dimension), either CHW or HWC
    input_batch_dim = model_config.max_batch_size > 0
    expected_input_dims = 3 + (1 if input_batch_dim else 0)
    if len(input_metadata.shape) != expected_input_dims:
        raise Exception(
            "expecting input to have {} dimensions, model '{}' input has {}".format(
                expected_input_dims, model_metadata.name, len(input_metadata.shape)
            )
        )

    if type(input_config.format) == str:
        FORMAT_ENUM_TO_INT = dict(mc.ModelInput.Format.items())
        input_config.format = FORMAT_ENUM_TO_INT[input_config.format]

    if (input_config.format != mc.ModelInput.FORMAT_NCHW) and (
        input_config.format != mc.ModelInput.FORMAT_NHWC
    ):
        raise Exception(
            "unexpected input format "
            + mc.ModelInput.Format.Name(input_config.format)
            + ", expecting "
            + mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NCHW)
            + " or "
            + mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NHWC)
        )

    if input_config.format == mc.ModelInput.FORMAT_NHWC:
        h = input_metadata.shape[1 if input_batch_dim else 0]
        w = input_metadata.shape[2 if input_batch_dim else 1]
        c = input_metadata.shape[3 if input_batch_dim else 2]
    else:
        c = input_metadata.shape[1 if input_batch_dim else 0]
        h = input_metadata.shape[2 if input_batch_dim else 1]
        w = input_metadata.shape[3 if input_batch_dim else 2]

    return (
        model_config.max_batch_size,
        input_metadata.name,
        output_metadata.name,
        c,
        h,
        w,
        input_config.format,
        input_metadata.datatype,
    )


def preprocess(img, format, dtype, c, h, w, scaling, protocol):
    """
    Pre-process an image to meet the size, type and format
    requirements specified by the parameters.
    """

    if c == 1:
        sample_img = img.convert("L")
    else:
        sample_img = img.convert("RGB")

    resized_img = sample_img.resize((w, h), Image.BILINEAR)
    resized = np.array(resized_img)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]

    npdtype = triton_to_np_dtype(dtype)
    typed = resized.astype(npdtype)

    if scaling == "INCEPTION":
        scaled = (typed / 127.5) - 1
    elif scaling == "VGG":
        if c == 1:
            scaled = typed - np.asarray((128,), dtype=npdtype)
        else:
            scaled = typed - np.asarray((123, 117, 104), dtype=npdtype)
    elif scaling == "CONVNEXT":
        if c == 1:
            scaled = (typed / 127.5) - 1
        else:
            scaled = (typed / 127.5) - 1
            imagenet_mean = np.asarray((0.485, 0.456, 0.406), dtype=npdtype)
            imagenet_std = np.asarray((0.229, 0.224, 0.225), dtype=npdtype)
            scaled = np.asarray((scaled - imagenet_mean) / imagenet_std, dtype=npdtype)
    else:
        scaled = typed

    # Swap to CHW if necessary
    if format == mc.ModelInput.FORMAT_NCHW:
        ordered = np.transpose(scaled, (2, 0, 1))
    else:
        ordered = scaled

    # Channels are in RGB order. Currently model configuration data
    # doesn't provide any information as to other channel orderings
    # (like BGR) so we just assume RGB.
    return ordered


def postprocess(results, output_name, batch_size, batching):
    """
    Post-process results to show classifications.
    """
    final_predictions = []
    output_array = results.as_numpy(output_name)

    # Include special handling for non-batching models
    if output_array.dtype.type == np.object_:
        for results in output_array:
            prediction_indices = []
            prediction_classes = []
            prediction_scores = []
            for result in results:
                if type(result) == bytes:
                    result_string = result.decode("utf-8")
                    items = result_string.split(":")
                    prediction_scores.append(items[0])
                    prediction_indices.append(items[1])
                    prediction_classes.append(items[2])
                elif type(result) == string:
                    items = result.split(":")
                    prediction_scores.append(items[0])
                    prediction_indices.append(items[1])
                    prediction_classes.append(items[2])
                else:
                    print("type(result): ", type(result))
                    print("result: ", result)
                    sys.stderr(
                        "The model output format is not handled yet in the post processing fn!"
                    )
            index_with_max_score = np.argmax(np.asarray(prediction_scores))
            predicted_class_name = prediction_classes[index_with_max_score]
            predicted_class_score = prediction_scores[index_with_max_score]
            final_predictions.append((predicted_class_name, predicted_class_score))
    else:
        print("type(output_array): ", output_array.dtype.type)
        print("output_array: ", output_array)
        sys.stderr(
            "The model output format is not handled yet in the post processing fn!"
        )
        pass
    return final_predictions


def requestGenerator(
    batched_image_data,
    input_name,
    output_name,
    dtype,
    protocol,
    no_of_classes,
    model_name,
    model_version,
):
    """
    A generator function to generate requests
    Parameters
    ----------
    batched_image_data : np.ndarray
    input_name : str
    output_name : str
    dtype : type_obj
    protocol : str
    no_of_classes : int
    model_name : str
    model_version : str
    Returns
    ----------
    generator_obj
    """
    protocol = protocol.lower()

    if protocol == "grpc":
        client = grpcclient
    else:
        client = httpclient

    # Set the input data
    inputs = [client.InferInput(input_name, batched_image_data.shape, dtype)]
    inputs[0].set_data_from_numpy(batched_image_data)

    outputs = [client.InferRequestedOutput(output_name, class_count=no_of_classes)]

    yield inputs, outputs, model_name, model_version


def convert_http_metadata_config(_metadata, _config):
    _model_metadata = AttrDict(_metadata)
    _model_config = AttrDict(_config)

    return _model_metadata, _model_config


def classify_with_triton(
    model_name,
    images_generator,
    no_of_images,
    model_version="",
    no_of_classes=4,
    batch_size=1,
    server_url="localhost:8000",
    scaling="CONVNEXT",
    protocol="HTTP",
    verbose=False,
    async_flag=False,
    streaming=False,
):
    """
    A function to call the triton inference server
    to classify a given set of images, given a set of
    model params. Supports http, grpc protocols, dynamic batch size
    async requests, streaming requests, uses requests library
    Parameters
    ----------
    model_name : str
    images_generator : generator_obj
    no_of_images : int
    model_version : str
    no_of_classes : int
    batch_size : int
    server_url : str
    scaling : str
    protocol : str
    verbose : bool
    async_flag : bool
    streaming : bool
    Returns
    ----------
    generator_obj
    """

    if streaming and protocol.lower() != "grpc":
        raise Exception("Streaming is only allowed with gRPC protocol")
    try:
        if protocol.lower() == "grpc":
            # Create gRPC client for communicating with the server
            triton_client = grpcclient.InferenceServerClient(
                url=server_url, verbose=verbose
            )
        else:
            # Specify large enough concurrency to handle the
            # the number of requests.
            concurrency = 20 if async_flag else 1
            triton_client = httpclient.InferenceServerClient(
                url=server_url, verbose=verbose, concurrency=concurrency
            )
    except Exception as e:
        print("client creation failed: " + str(e))
        sys.exit(1)
    # Make sure the model matches our requirements, and get some
    # properties of the model that we need for preprocessing
    try:
        model_metadata = triton_client.get_model_metadata(
            model_name=model_name, model_version=model_version
        )
    except InferenceServerException as e:
        print("failed to retrieve the metadata: " + str(e))
        sys.exit(1)
    try:
        model_config = triton_client.get_model_config(
            model_name=model_name, model_version=model_version
        )
    except InferenceServerException as e:
        print("failed to retrieve the config: " + str(e))
        sys.exit(1)
    if protocol.lower() == "grpc":
        model_config = model_config.config
    else:
        model_metadata, model_config = convert_http_metadata_config(
            model_metadata, model_config
        )

    max_batch_size, input_name, output_name, c, h, w, format, dtype = parse_model(
        model_metadata, model_config
    )

    responses = []
    user_data = UserData()

    # Holds the handles to the ongoing HTTP async requests.
    async_requests = []

    sent_count = 0
    image_index = 0
    if streaming:
        triton_client.start_stream(partial(completion_callback, user_data))
    for j in range(0, no_of_images, batch_size):
        repeated_image_data = []
        no_of_images_in_this_request = min(batch_size, no_of_images - image_index)
        for idx in range(no_of_images_in_this_request):
            # You may need to convert the color if it is not Pillow style RGB format
            pil_image = next(images_generator)
            image_index += 1
            repeated_image_data.append(
                preprocess(pil_image, format, dtype, c, h, w, scaling, protocol.lower())
            )

        if max_batch_size > 0:
            batched_image_data = np.stack(repeated_image_data, axis=0)
        else:
            batched_image_data = repeated_image_data[0]

        # Send request
        try:
            for inputs, outputs, model_name, model_version in requestGenerator(
                batched_image_data,
                input_name,
                output_name,
                dtype,
                protocol,
                no_of_classes,
                model_name,
                model_version,
            ):
                sent_count += 1
                if streaming:
                    triton_client.async_stream_infer(
                        model_name,
                        inputs,
                        request_id=str(sent_count),
                        model_version=model_version,
                        outputs=outputs,
                    )
                elif async_flag:
                    if protocol.lower() == "grpc":
                        triton_client.async_infer(
                            model_name,
                            inputs,
                            partial(completion_callback, user_data),
                            request_id=str(sent_count),
                            model_version=model_version,
                            outputs=outputs,
                        )
                    else:
                        async_requests.append(
                            triton_client.async_infer(
                                model_name,
                                inputs,
                                request_id=str(sent_count),
                                model_version=model_version,
                                outputs=outputs,
                            )
                        )
                else:
                    responses.append(
                        triton_client.infer(
                            model_name,
                            inputs,
                            request_id=str(sent_count),
                            model_version=model_version,
                            outputs=outputs,
                        )
                    )

        except InferenceServerException as e:
            print("inference failed: " + str(e))
            if streaming:
                triton_client.stop_stream()
            sys.exit(1)

    if streaming:
        triton_client.stop_stream()

    if protocol.lower() == "grpc":
        if streaming or async_flag:
            processed_count = 0
            while processed_count < sent_count:
                (results, error) = user_data._completed_requests.get()
                processed_count += 1
                if error is not None:
                    print("inference failed: " + str(error))
                    sys.exit(1)
                responses.append(results)
    else:
        if async_flag:
            # Collect results from the ongoing async requests
            # for HTTP Async requests.
            for async_request in async_requests:
                responses.append(async_request.get_result())
    all_final_predictions = []
    for response in responses:
        # if protocol.lower() == "grpc":
        #     this_id = response.get_response().id
        # else:
        #     this_id = response.get_response()["id"]
        # print("Request {}, batch size {}".format(this_id, batch_size))
        final_predictions = postprocess(
            response, output_name, batch_size, max_batch_size > 0
        )
        # print ('length of final_predictions: ', len(final_predictions))
        all_final_predictions.extend(final_predictions)
    try:
        assert no_of_images == len(all_final_predictions)
    except AssertionError as AE:
        print("no_of_images: ", no_of_images)
        print("len(all_final_predictions): ", len(all_final_predictions))
        raise AE
    return all_final_predictions


def unit_test_gen():
    for i in range(30):
        yield Image.open(
            "/home/aboggaram/data/pgf_healthy_classification_data_March_7_2022/val/empty/pgf-9-18-roboarc117-1-119-18-43_970_26_55.jpg"
        )


# unit_test()
def unit_test():
    unit_test_gen_obj = unit_test_gen()
    all_final_predictions = classify_with_triton(
        "convnext_onnx",
        unit_test_gen_obj,
        no_of_images=30,
        batch_size=16,
    )
    print("length of all_final_predictions: ", len(all_final_predictions))
    print(all_final_predictions[0])
