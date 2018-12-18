import os 
import tensorflow as tf
from tensorflow_transform.tf_metadata import dataset_metadata, metadata_io,dataset_schema 
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow.metrics import recall,precision

# CONSTANTS
BATCH_SIZE=2 # Hyperparamter
SENTENCE_LEN=30 # Pre processing paramter
WORD_LEN=10  # Pre processing paramter
CHAR_SIZE=54  # Number of chars, Pre processing paramter
EMBED_DIM_CHAR=int(CHAR_SIZE**(1/4))*8 # Hyperparamter
DEFAULT_WORD_VALUE = 0 # Pre processing paramter
VOCAB_SIZE=1e4  # Preprocessing paramter
EMBED_DIM_WORD=int(VOCAB_SIZE**(1/4))*8 # Hyperparamter
CHAR_HIDDEN_SIZE=10 # Hyperparamter
HIDDEN_SIZE=20 # Hyperparamter
NUM_CLASSES=9


tf.reset_default_graph()
def transform_metadata(folder="gs://named_entity_recognition/beam/"):
    """Read the transform metadata"""
    transformed_metadata = metadata_io.read_metadata(
    os.path.join(
        folder, transform_fn_io.TRANSFORMED_METADATA_DIR
        )
    )
    transformed_feature_spec = transformed_metadata.schema.as_feature_spec()
    return transformed_feature_spec


def train_input(train_folder=None,model_dir_beam=None,batch_size=None):
    """Function to generate the tinput data"""
    transformed_feature_spec = transform_metadata()
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=os.path.join(train_folder,"TEST*"),
        batch_size=BATCH_SIZE,
        features=transformed_feature_spec,
        reader=tf.data.TFRecordDataset,
        shuffle=True
        )
    transformed_features = dataset.make_one_shot_iterator().get_next()
    ## Need to change here later to get all the features
    transformed_labels = {key: value for (key, value) in transformed_features.items() if key in ["labels"]}
    transformed_features = {key: value for (key, value) in transformed_features.items() if key not in ["labels"]}
    
    return transformed_features, transformed_labels


def eval_input(input=None):
    transformed_feature_spec = transform_metadata()
    pass


def serve_input(input=None):
    pass


def model_fn(features,labels,mode,params):
    """Function to genereate the input data"""
    # Embedd 
    input_char = tf.reshape(features["chars"],[BATCH_SIZE,SENTENCE_LEN,WORD_LEN])
    char_embedding = tf.contrib.layers.embed_sequence(
        input_char, vocab_size=CHAR_SIZE, embed_dim=EMBED_DIM_CHAR)
    shape_char_embeddings = tf.shape(char_embedding)
    char_embeddings = tf.reshape(char_embedding, shape=[-1, shape_char_embeddings[-2], shape_char_embeddings[-1]])
    
    word_lengths = tf.reshape(features['chars_in_word'],[-1])

    input_word = tf.sparse.to_dense(features["word_representation"],default_value=DEFAULT_WORD_VALUE)
    paddings = tf.constant([[0, 0,], [0, SENTENCE_LEN]])
    ## Here we do the padding
    input_word = tf.pad(input_word, paddings, "CONSTANT",constant_values=DEFAULT_WORD_VALUE)
    input_word= tf.slice(input_word, [0, 0], [BATCH_SIZE, SENTENCE_LEN])

    input_word = tf.reshape(input_word,[BATCH_SIZE,SENTENCE_LEN])
    word_embedding = tf.contrib.layers.embed_sequence(
        input_word, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM_WORD)

    cell_fw1 = tf.contrib.rnn.LSTMCell(CHAR_HIDDEN_SIZE, state_is_tuple=True)
    cell_bw1 = tf.contrib.rnn.LSTMCell(CHAR_HIDDEN_SIZE, state_is_tuple=True)

    _, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw1,
        cell_bw1, char_embeddings, sequence_length=word_lengths,
        dtype=tf.float32,scope="bidirectional_rnn_1")

    # CONCAT THE FORWARD AND BACKWARD PASSED FEATURES
    output = tf.concat([output_fw, output_bw], axis=-1)

    # shape = (batch, sentence, 2 x char_hidden_size)
    char_rep = tf.reshape(output, shape=[-1, shape_char_embeddings[1], 2*CHAR_HIDDEN_SIZE])

    # shape = (batch, sentence, 2 x char_hidden_size + word_vector_size)
    word_embeddings = tf.concat([word_embedding, char_rep], axis=-1)
    cell_fw2 = tf.contrib.rnn.LSTMCell(HIDDEN_SIZE)
    cell_bw2 = tf.contrib.rnn.LSTMCell(HIDDEN_SIZE)
    sequence_length= tf.reshape(features['sentence_length'],[BATCH_SIZE])
    (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw2,
        cell_bw2, word_embeddings, sequence_length=sequence_length,
        dtype=tf.float32,scope="bidirectional_rnn_2")
    context_rep = tf.concat([output_fw, output_bw], axis=-1)

    

    ntime_steps = tf.shape(context_rep)[1]
    context_rep_flat = tf.reshape(context_rep, [-1, 2*HIDDEN_SIZE])
    preds = tf.layers.dense(inputs = context_rep_flat,units=NUM_CLASSES)
    logits = tf.reshape(preds, [-1, ntime_steps, NUM_CLASSES])

    crf_params = tf.get_variable("crf", [NUM_CLASSES, NUM_CLASSES], dtype=tf.float32)
    pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, sequence_length)

    LOOK_UP_TABLE="FILE_NAME_TO_LOOP_UP_VOCAB"

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Predictions
        # GET THE WORD FORM THE ID, LINK THIS WITH THE BEAM TRANSFORM VOCAB
        reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(
            LOOK_UP_TABLE)
        pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))
        predictions = {
            'pred_ids': pred_ids,
            'tags': pred_strings
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:

        labels = tf.reshape(labels['labels'],[BATCH_SIZE,SENTENCE_LEN])
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            logits, labels, sequence_length, crf_params)
        loss = tf.reduce_mean(-log_likelihood)

        # Metrics
        # AWESOME THIS IS GREAT TO USE THIS MASK HERE
        # WILL SOLVE A LOT OF PROBLEMS, OTHERWISE
        # GOOD LEARNING AS WELL
        #weights = tf.sequence_mask(sequence_length)
        weights = tf.sequence_mask(sequence_length,maxlen=SENTENCE_LEN)
        nbr = tf.constant(NUM_CLASSES)
        # NEED TO FIX THIS AND UNDERSTAND IT ASWELL
        metrics = {
            'acc': tf.metrics.accuracy(labels, pred_ids, weights),
            #'precision': precision(labels, pred_ids, nbr, weights),
            #'recall': recall(labels, pred_ids, NUM_CLASSES, weights),
            #'f1': f1(labels, pred_ids, NUM_CLASSES, weights),
        }
        # ADD THEM ALL TO THE GRAPH
        #for metric_name, op in metrics.items():
        #    tf.summary.scalar(metric_name, op[1])
        
        # WHAT WILL HAPPEN IF WE HAVE EVAL MODE
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        # WHAT WILL HAPPEN IF WE HAVE TRAIN? 
        elif mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer().minimize(
                loss, global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op)






















