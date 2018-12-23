import argparse
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.contrib.training.python.training import hparam
from trainer.model import model_fn,train_input,eval_input

def model_setup(params=None):
	"""Model set up function returns an custom estimator"""
	run_config = tf.estimator.RunConfig(save_checkpoints_secs = 120, 
	                                    keep_checkpoint_max = 3)
	# The model is in the model file
	cnn_classifier = tf.estimator.Estimator(model_fn=model_fn,
	                                        model_dir=params.job_dir,
	                                        params=hparams,
	                                        config = run_config)
	return cnn_classifier

def main(hparams):
    tf.logging.set_verbosity(tf.logging.INFO)
    classifier=model_setup(params = hparams)
    hook = tf.contrib.estimator.stop_if_no_increase_hook(
        classifier, 'f1', 500, min_steps=8000, run_every_secs=120)
    #train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
    train_spec = tf.estimator.TrainSpec(input_fn=lambda:train_input(
			train_folder = hparams.train_folder,
			model_dir_beam = hparams.model_dir_beam,
			batch_size = int(hparams.batch_size)
			),
		max_steps=int(hparams.max_steps),
        hooks=[hook]
		)

    #tf_transform_beam = tft.TFTransformOutput(hparams.model_dir_beam)
    #serving_input_fn = trainer.model.serve_input(tf_transform_beam)
    #exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda:eval_input(
			train_folder = hparams.eval_folder,
			model_dir_beam = hparams.model_dir_beam,
			batch_size = int(hparams.batch_size)
			),
		start_delay_secs=30, 
		throttle_secs=40,
		#exporters=exporter
		)
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
    pass



if __name__=="__main__":
		parser = argparse.ArgumentParser()
		parser.add_argument(
		'--job-dir',
		help='GCS location to write checkpoints and export models',
		default="gs://named_entity_recognition/ml_engine")
		parser.add_argument(
		'--max-steps',
		help='max number of traning steps',
		default=5000)
		parser.add_argument(
		'--model-dir-beam',
		help='path to the model dir, where the beam job stored the outputs',
		default="gs://named_entity_recognition/beam")
		parser.add_argument(
		'--train-folder',
		help='path to the train folder',
		default="gs://named_entity_recognition/beam/train")
		parser.add_argument(
		'--eval-folder',
		help='path to the eval folder',
		default="gs://named_entity_recognition/beam/train")
		parser.add_argument(
		'--batch-size',
		help='the batch size',
		default=100)
		parser.add_argument(
		'--sentence-len',
		help='the max length of a sentence',
		default=54)
		parser.add_argument(
		'--word-len',
		help='the max length of a word',
		default=30)
		parser.add_argument(
		'--char-size',
		help='number of characters used, based upon settings in preprocessing' ,
		default=54)
		parser.add_argument(
		'--embedd-char-size',
		help='the size of the char embeddings',
		default=26)
		parser.add_argument(
		'--default-word-value',
		help='what should be used during padding as defualt value',
		default=0)
		parser.add_argument(
		'--vocab_size',
		help='this is based upon the preprocessing job',
		default=1e4)
		parser.add_argument(
		'--embedd-word-size',
		help='how big should the word embedding be',
		default=1e4**(1/4))
		parser.add_argument(
		'--char-hidden-size',
		help="The size of the hidden units for the chars",
		default=10)
		parser.add_argument(
		'--hidden-size',
		help='the hidden size of the main lstm units',
		default=20)
		parser.add_argument(
		'--num_classes',
		help='the number of classes',
		default=9)
		args, _ = parser.parse_known_args()
		tf.logging.set_verbosity(tf.logging.INFO)
		hparams = hparam.HParams(**args.__dict__)
		main(hparams=hparams)

