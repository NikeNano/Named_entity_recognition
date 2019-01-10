import os 
import tensorflow as tf
from tensorflow_transform.tf_metadata import dataset_metadata, metadata_io,dataset_schema 
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow.metrics import accuracy
from trainer.metrics import precision,recall,f1


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
        file_pattern=os.path.join(train_folder,"TRAIN*"),
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=tf.data.TFRecordDataset,
        shuffle=True
        )
    transformed_features = dataset.make_one_shot_iterator().get_next()
    ## Need to change here later to get all the features
    transformed_labels = {key: value for (key, value) in transformed_features.items() if key in ["labels"]}
    transformed_features = {key: value for (key, value) in transformed_features.items() if key not in ["labels"]}    
    return transformed_features, transformed_labels


def eval_input(train_folder=None,model_dir_beam=None,batch_size=None):
    """Function to generate the tinput data"""
    transformed_feature_spec = transform_metadata()
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=os.path.join(train_folder,"TEST*"),
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=tf.data.TFRecordDataset,
        shuffle=True
        )
    transformed_features = dataset.make_one_shot_iterator().get_next()
    ## Need to change here later to get all the features
    transformed_labels = {key: value for (key, value) in transformed_features.items() if key in ["labels"]}
    transformed_features = {key: value for (key, value) in transformed_features.items() if key not in ["labels"]}
    return transformed_features, transformed_labels


def build_serving_input_fn(tf_transform_beam,params):
    """Creates an input function reading from raw data.
    Args:
    Returns:
        The serving input function.
    """
    TRANSFORM_INPUT_SCHEMA = dataset_schema.from_feature_spec({
        'text': tf.FixedLenFeature(shape=[], dtype=tf.string),
        "chars" : tf.FixedLenFeature(shape=[int(params.sentence_len),int(params.word_len)], dtype=tf.int64),
        'sentence_length': tf.FixedLenFeature(shape=[], dtype=tf.int64),
        'chars_in_word': tf.FixedLenFeature(shape=[int(params.sentence_len)], dtype=tf.int64),
        })
    raw_feature_spec = TRANSFORM_INPUT_SCHEMA.as_feature_spec()
    def serving_input_fn():
            """
            Receiver function that converts raw features into transformed features
            :return: ServingInputReceiver
            """
            raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
                raw_feature_spec)
            raw_features, _, _ = raw_input_fn()
            transformed_features = tf_transform_beam.transform_raw_features(
                raw_features)
            return tf.estimator.export.ServingInputReceiver(
                transformed_features, raw_features)
    return serving_input_fn


def model_fn(features,labels,mode,params):
    """Function to genereate the input data"""
    # Set up paramters
    #BATCH_SIZE=int(params.batch_size) 
    SENTENCE_LEN=int(params.sentence_len)
    WORD_LEN=int(params.word_len)
    CHAR_SIZE=int(params.char_size)
    EMBED_DIM_CHAR=int(params.embedd_char_size)
    DEFAULT_WORD_VALUE = int(params.default_word_value)
    VOCAB_SIZE=int(params.vocab_size)
    EMBED_DIM_WORD=int(params.embedd_word_size)
    CHAR_HIDDEN_SIZE=int(params.char_hidden_size)
    HIDDEN_SIZE=int(params.hidden_size)
    NUM_CLASSES=int(params.num_classes)
    INDICIES =list(range(1,9))
    LOOK_UP_TABLE=str(params.lookup_file)
    BATCH_SIZE = tf.shape(features["chars"])[0]
    input_char = tf.reshape(features["chars"],[BATCH_SIZE,SENTENCE_LEN,WORD_LEN])
    char_embeddings = tf.contrib.layers.embed_sequence(
        input_char, vocab_size=CHAR_SIZE, embed_dim=EMBED_DIM_CHAR)
    shape_char_embeddings = tf.shape(char_embeddings)
    #char_embeddings = tf.reshape(char_embeddings, shape=[-1, shape_char_embeddings[-2], shape_char_embeddings[-1]])
    char_embeddings = tf.reshape(char_embeddings, shape=[-1, WORD_LEN, EMBED_DIM_CHAR])
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
    output = tf.concat([output_fw, output_bw], axis=-1)
    char_rep = tf.reshape(output, shape=[-1, shape_char_embeddings[1], 2*CHAR_HIDDEN_SIZE])
    word_embeddings = tf.concat([word_embedding, char_rep], axis=-1)
    cell_fw2 = tf.contrib.rnn.LSTMCell(HIDDEN_SIZE)
    cell_bw2 = tf.contrib.rnn.LSTMCell(HIDDEN_SIZE)
    sequence_length= tf.reshape(features['sentence_length'],[BATCH_SIZE])
    #hook1 =  tf.train.LoggingTensorHook({"sequence_length:": sequence_length},
    #                           every_n_iter=1)
    #hook2 =  tf.train.LoggingTensorHook({"word_embedding:": word_embeddings},
    #                           every_n_iter=1)                           
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
    if mode == tf.estimator.ModeKeys.PREDICT:
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
        weights = tf.sequence_mask(sequence_length,maxlen=SENTENCE_LEN)
        metrics = {
            'acc': accuracy(labels, pred_ids, weights),
            'precision': precision(labels, pred_ids, NUM_CLASSES, INDICIES, weights),
            'recall': recall(labels, pred_ids, NUM_CLASSES, INDICIES, weights),
            'f1': f1(labels, pred_ids, NUM_CLASSES, INDICIES, weights),
            #'mean per class accuracy': mean_per_class_accuracy(labels, pred_ids, weights)
        }
        hook_pred_ids =  tf.train.LoggingTensorHook({"pred_ids:": pred_ids},
                               every_n_iter=1)
        # ADD THEM ALL TO THE GRAPH
        for metric_name, op in metrics.items():
            tf.summary.scalar(metric_name, op[1])
        
        # WHAT WILL HAPPEN IF WE HAVE EVAL MODE
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        # WHAT WILL HAPPEN IF WE HAVE TRAIN? 
        elif mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer().minimize(
                loss, global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op)#,training_hooks=[hook_shape])#,training_hooks=[hook1,hook2])






















