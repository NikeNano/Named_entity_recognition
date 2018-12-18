import argparse
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.contrib.training.python.training import hparam

from trainer.model import model_fn,train_input

def model_setup(params=None):
	"""Model set up function returns an custom estimator"""
	run_config = tf.estimator.RunConfig(save_checkpoints_secs = 10, 
	                                    keep_checkpoint_max = 3)
	# The model is in the model file
	cnn_classifier = tf.estimator.Estimator(model_fn=model_fn,
	                                        model_dir=params.job_dir,
	                                        params=params,
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
    
    tf_transform_beam = tft.TFTransformOutput(hparams.model_dir_beam)
    #serving_input_fn = trainer.model.serve_input(tf_transform_beam)
    #exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda:train_input(
			train_folder = hparams.train_folder,
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
		'--input-filename',
		help='GCS file or local paths to data',
		nargs='+', # makes it a list :)
		default='gs://named_entity_recognition/beam')
    parser.add_argument(
		'--model-dir',
		help='where to ',
		nargs='+', # makes it a list :)
		default='gs://named_entity_recognition/beam')
    parser.add_argument(
		'--job-dir',
		help='GCS location to write checkpoints and export models',
		default="gs://named_entity_recognition/ml_engine")
    parser.add_argument(
		'--max-steps',
		help='max number of traning steps',
		default=100)
    parser.add_argument(
		'--model-dir-beam',
		help='path to the model dir, where the beam job stored the outputs',
		default="gs://named_entity_recognition/beam")
    parser.add_argument(
		'--train-folder',
		help='path to the train folder',
		default="gs://named_entity_recognition/beam")
    parser.add_argument(
		'--eval-folder',
		help='path to the eval folder',
		default="gs://named_entity_recognition/beam")
    parser.add_argument(
		'--batch-size',
		help='the batch size',
		default=100)
    args, _ = parser.parse_known_args()
    tf.logging.set_verbosity(tf.logging.INFO)
    hparams = hparam.HParams(**args.__dict__)
    main(hparams=hparams)

