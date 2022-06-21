import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile
from tensorflow.python.tools import strip_unused_lib


def check_tensorflow_version():
    tf_version = tf.__version__
    major_version = int(tf_version.split(".")[0])

    assert major_version == 1, "The TensorFlow version required for this script is tf1.x. " \
                               "TensorFlow 1.15 is recommended."


def load_saved_model(path):
    the_graph = tf.Graph()
    with tf.Session(graph=the_graph) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], path)
    return the_graph


saved_model_path = "ssdlite_mobilenet_v2_coco_2018_05_09/saved_model"

the_graph = load_saved_model(saved_model_path)

input_node = "Preprocessor/sub"
bbox_output_node = "concat"
class_output_node = "Postprocessor/convert_scores"


def optimize_graph(graph):
    gdef = strip_unused_lib.strip_unused(
        input_graph_def=graph.as_graph_def(),
        input_node_names=[input_node],
        output_node_names=[bbox_output_node, class_output_node],
        placeholder_type_enum=dtypes.float32.as_datatype_enum)

    return gdef


opt_gdef = optimize_graph(the_graph)

with gfile.GFile("saved_model.pb", "wb") as f:
    f.write(opt_gdef.SerializeToString())
