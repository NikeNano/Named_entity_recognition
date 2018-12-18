import argparse
import logging
import os
import sys
import tempfile
import random
import json
from datetime import datetime

import apache_beam as beam
import tensorflow as tf
import tensorflow_transform as tft
from string import ascii_lowercase
from apache_beam.io import tfrecordio
from tensorflow_transform import coders
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.tf_metadata import dataset_metadata, metadata_io,dataset_schema 

global someVar
someVar = 55
DELIMITERS_WORDS= ' '
#DELIMITERS_WORDS = '.,!?() '
MAX_WORD_LEN=10
MAX_SENTENCE_LEN=30

TRAIN_DATA_SCHEMA = dataset_schema.from_feature_spec({
    'id': tf.FixedLenFeature(shape=[], dtype=tf.float32),
    'text': tf.FixedLenFeature(shape=[], dtype=tf.string),
    'labels': tf.FixedLenFeature(shape=[MAX_SENTENCE_LEN], dtype=tf.int64),
    'label': tf.FixedLenFeature(shape=[], dtype=tf.string),
    "chars" : tf.FixedLenFeature(shape=[MAX_SENTENCE_LEN,MAX_WORD_LEN], dtype=tf.int64),
    'label_length': tf.FixedLenFeature(shape=[], dtype=tf.int64),
    'sentence_length': tf.FixedLenFeature(shape=[], dtype=tf.int64),
    'chars_in_word': tf.FixedLenFeature(shape=[MAX_SENTENCE_LEN], dtype=tf.int64),
})


train_metadata = dataset_metadata.DatasetMetadata(schema=TRAIN_DATA_SCHEMA)
OUTPUT_FILE_TRAIN="train_records.tfrecords"
MAPPING = {a:index for index,a in enumerate(ascii_lowercase + ascii_lowercase.upper())}
UNKONWN = len(MAPPING)
PADD_VALUE = UNKONWN+1

LABEL_MAPPING = {"B-LOC":0,"I-LOC":1,"B-PER":2,"I-PER":3,
    "B-ORG":4,"I-ORG":5,"B-MISC":6,"I-MISC":7,"O":8
    }

# Look here for support and extra guide
# https://github.com/guillaumegenthial/tf_ner/blob/master/models/chars_lstm_lstm_crf/main.py

def get_cloud_pipeline_options():
    """Get apache beam pipeline options to run with Dataflow on the cloud
    Args:
        project (str): GCP project to which job will be submitted
    Returns:
        beam.pipeline.PipelineOptions
    """
    options = {
        'runner': 'DataflowRunner',
        'job_name': ('preprocessdigitaltwin-{}'.format(
            datetime.now().strftime('%Y%m%d%H%M%S'))),
        'staging_location': "gs://named_entity_recognition/binaries/",
        'temp_location':  "gs://named_entity_recognition/tmp/",
        'project': "iotpubsub-1536350750202",
        'region': 'europe-west1',
        'zone': 'europe-west1-b',
        'autoscaling_algorithm': 'THROUGHPUT_BASED',
        'save_main_session': True,
        'setup_file': './setup.py',
    }
    return beam.pipeline.PipelineOptions(flags=[], **options)


def printy(x):
    print(x)

class PadList(list):
    """ The super padding list used for padding data
    """
    def inner_pad(self, pad_length, pad_value=0):
        """Do inner padding of the list
        
        Paramters:
            padd_length -- How long should the list be
            padd_value -- What value should be used for the padding
        
        Return:
            self -- the list 
        """
        nbr_pad = pad_length - len(self)
        if nbr_pad>0:
            self = self + [pad_value] * nbr_pad
        else:
            self=self[:pad_length]
        return self
    
    
    def outer_pad(self,padded_list_length,pad_length,pad_value=0):
        """
        Out padding of a list e.g append a list to a list. 
        Args:
            padded_list_length -- how long should the appended list be
            pad_lenght -- how long should the list be e.g how much should we append
            padd_value -- What should the appended list have as values
            
        """
        nbr_pad = pad_length-len(self)
        if nbr_pad > 0:
            for _ in range(nbr_pad):
                self.append([pad_value] * padded_list_length)
        else:
            self = self[:pad_length]
        return self


class Set_Key_Value(beam.DoFn):
    """Creat a key valye pair to later use for CoGroupByKey"""
    def process(self,element):
        """ Process function for the ParDo
        Paramters: 
            element -- the input element
        Returns list with tuple key value pair
        """
        key= int(element.split(" ")[0])
        value=" ".join(element.split(" ")[1:])
        return [(key,value)]


def preprocessing_fn(inputs):
    """ This is the preprocessing functions use by the tensorflow transform 
    Paramters:
        inputs -- the tensorflow parset input tensors in a dict, defined by the metadata input
    Returns:
        inputs -- dict wit the now appended output values, the added word representation is a sparse tensor
    """
    words = tf.string_split(inputs['text'],DELIMITERS_WORDS)
    word_representation = tft.compute_and_apply_vocabulary(words,default_value=0,top_k=10000)
    inputs["word_representation"] = word_representation

    return inputs

class Char_parser(beam.DoFn):
    """ Function will concert chars to integers and padd it so each tensor have same lenghts """
    def process(self,element):
        """ The pardo processing. 
            Paramters:
            element -- the input data, is a dict 
        Return:
            list(element) -- the output is a dict in a list
        """
        text = element["text"]
        mapped=PadList()
        for word in text.split(" "):
            tmp_mapped=PadList()
            for char in word:
                try:
                    tmp_mapped.append(MAPPING[char.strip()])
                except:
                    tmp_mapped.append(UNKONWN)
            tmp_mapped=tmp_mapped.inner_pad(MAX_WORD_LEN,PADD_VALUE)
            mapped.append(tmp_mapped)
        mapped = mapped.outer_pad(padded_list_length=MAX_WORD_LEN,pad_length=MAX_SENTENCE_LEN,pad_value=PADD_VALUE)
        element["chars"] = mapped
        return [element]

class Label_parser(beam.DoFn):
    """Function to parse the labels"""
    def process(self,element):
        ### HERE FIX THE LABELS AND PARSE THEM CORRECTLY TO BREAK IT UP TO LIST
        labels = element["label"]
        labels = labels.split(" ")
        labels = [LABEL_MAPPING[label] for label in labels]
        labels=PadList(labels)
        element["labels"] =labels.inner_pad(MAX_SENTENCE_LEN, max(LABEL_MAPPING.values()))

        label_length = len(element["label"].split())
        sentence_length = len(element["text"].split())
        element["sentence_length"] = sentence_length
        element["label_length"] = label_length
        word_lengths  = PadList([len(word) if len(word)<MAX_WORD_LEN else MAX_WORD_LEN for word in element["text"].split()])
        element["chars_in_word"] = word_lengths.inner_pad(MAX_SENTENCE_LEN,0)

        return [element]


class Reshape_data(beam.DoFn):
    """Funxtion to reshape"""
    def process(self,element):
        key = element[0]
        # This will result in that we remove unknonw 
        text = str(element[1]["text"][0].encode('ascii','ignore'))
        labels = str(element[1]["labels"][0])#.encode('ascii','ignore'))
        return [{"id":key, "text":text,"label":labels}]


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cloud', type=str, help='y' )
    args = parser.parse_args(args=None) # Parse the arguments 
    
    if args.cloud=="y":
        pipeline_options = get_cloud_pipeline_options()
    else:
        pipeline_options = None
    p = beam.Pipeline(options=pipeline_options)
    with beam_impl.Context(temp_dir="gs://named_entity_recognition/beam/"):
        filename_data = "gs://named_entity_recognition/data/text_new.txt"
        filename_labels = "gs://named_entity_recognition/data/labels_new.txt"
        raw_text = (p | 'Read input data' >> beam.io.ReadFromText(filename_data)
            | 'Set key value data' >> beam.ParDo(Set_Key_Value())
            )
        raw_labels = (p | 'ReadInputLabels' >> beam.io.ReadFromText(filename_labels)  
            | 'Set key value labels' >> beam.ParDo(Set_Key_Value())
            )
        train_data = ({'text': raw_text,'labels': raw_labels}
            | "merge the labels and the data" >> beam.CoGroupByKey()
            | "reshape " >> beam.ParDo(Reshape_data())
            | "encode the chars" >> beam.ParDo(Char_parser())
            | "convert the " >> beam.ParDo(Label_parser())
            )

        train_dataset = (train_data, train_metadata)
        transformed_dataset, transform_fn = (train_dataset
                                              | 'AnalyzeAndTransform' >> beam_impl.AnalyzeAndTransformDataset(
                     preprocessing_fn))
        
        transformed_data, transformed_metadata = transformed_dataset

        transformed_data_coder = tft.coders.ExampleProtoCoder(
            transformed_metadata.schema)

        _ = (transformed_data
             | 'EncodeDataTrain' >> beam.Map(transformed_data_coder.encode)
             | 'WriteDataTrain' >> tfrecordio.WriteToTFRecord(os.path.join("gs://named_entity_recognition/beam/","TEST")))
        
        _ = (transform_fn
             | "WriteTransformFn" >>
             transform_fn_io.WriteTransformFn("gs://named_entity_recognition/beam/"))

        # Add the flow for the test data!
        # Then it is time to start modelling :) 
        # This should be fun!
        # TO DO 1
        # Fix the test data flow
        # Continue on to do the modellinng of the data instead
        # https://cs230-stanford.github.io/pytorch-nlp.html


        p.run().wait_until_finish()
if __name__=="__main__":
    main()