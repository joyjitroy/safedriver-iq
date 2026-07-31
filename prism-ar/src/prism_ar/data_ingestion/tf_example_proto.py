"""Minimal protobuf parser for tf.train.Example (no TensorFlow).

Used by the Waymo WOMD loader to parse TFRecord files with the `tfrecord` package.
"""
from __future__ import annotations

import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


def _build_example_descriptor():
    """Build a dynamic protobuf descriptor for tf.train.Example."""
    file_desc = descriptor_pb2.FileDescriptorProto()
    file_desc.name = "tf_example.proto"
    file_desc.package = "tensorflow"

    # BytesList
    bytes_list = file_desc.message_type.add()
    bytes_list.name = "BytesList"
    value_field = bytes_list.field.add()
    value_field.name = "value"
    value_field.number = 1
    value_field.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    value_field.type = descriptor_pb2.FieldDescriptorProto.TYPE_BYTES

    # FloatList
    float_list = file_desc.message_type.add()
    float_list.name = "FloatList"
    f_field = float_list.field.add()
    f_field.name = "value"
    f_field.number = 1
    f_field.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    f_field.type = descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT

    # Int64List
    int64_list = file_desc.message_type.add()
    int64_list.name = "Int64List"
    i_field = int64_list.field.add()
    i_field.name = "value"
    i_field.number = 1
    i_field.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    i_field.type = descriptor_pb2.FieldDescriptorProto.TYPE_INT64

    # Feature
    feature = file_desc.message_type.add()
    feature.name = "Feature"
    bf = feature.field.add()
    bf.name = "bytes_list"
    bf.number = 1
    bf.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    bf.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    bf.type_name = ".tensorflow.BytesList"
    ff = feature.field.add()
    ff.name = "float_list"
    ff.number = 2
    ff.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    ff.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    ff.type_name = ".tensorflow.FloatList"
    iff = feature.field.add()
    iff.name = "int64_list"
    iff.number = 3
    iff.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    iff.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    iff.type_name = ".tensorflow.Int64List"

    # Features
    features = file_desc.message_type.add()
    features.name = "Features"
    map_field = features.field.add()
    map_field.name = "feature"
    map_field.number = 1
    map_field.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    map_field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    map_field.type_name = ".tensorflow.Features.FeatureEntry"
    # Add the map entry message
    entry = features.nested_type.add()
    entry.name = "FeatureEntry"
    entry.options.map_entry = True
    k = entry.field.add()
    k.name = "key"
    k.number = 1
    k.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    k.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING
    v = entry.field.add()
    v.name = "value"
    v.number = 2
    v.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    v.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    v.type_name = ".tensorflow.Feature"
    map_field.type_name = ".tensorflow.Features.FeatureEntry"

    # Example
    example = file_desc.message_type.add()
    example.name = "Example"
    feat_field = example.field.add()
    feat_field.name = "features"
    feat_field.number = 1
    feat_field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    feat_field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    feat_field.type_name = ".tensorflow.Features"

    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_desc)
    try:
        factory = message_factory.GetMessageClassesForFiles([file_desc.name], pool)
    except TypeError:
        # Older protobuf API
        factory = message_factory.GetMessageClassesForFiles([file_desc.name])
    return factory["tensorflow.Example"]


_Example = _build_example_descriptor()


def parse_example(raw_bytes: bytes) -> dict:
    """Parse a tf.train.Example protobuf into a dictionary of numpy arrays."""
    example = _Example()
    example.ParseFromString(raw_bytes)
    result = {}
    for key, feature in example.features.feature.items():
        if feature.HasField("bytes_list"):
            result[key] = np.array([v for v in feature.bytes_list.value], dtype=object)
        elif feature.HasField("float_list"):
            result[key] = np.array(feature.float_list.value, dtype=np.float32)
        elif feature.HasField("int64_list"):
            result[key] = np.array(feature.int64_list.value, dtype=np.int64)
    return result
