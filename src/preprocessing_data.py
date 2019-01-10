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

DELIMITERS_WORDS=' '
MAX_WORD_LEN=15
MAX_SENTENCE_LEN=25

# IT IS EXTREMELY COSTFUL TO HAVE TO LONG WORDS/SENTENCES

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
PADD_VALUE_CHAR = UNKONWN+1

LABEL_MAPPING = {"O":0,"B-LOC":1,"I-LOC":2,"B-PER":3,"I-PER":4,
    "B-ORG":5,"I-ORG":6,"B-MISC":7,"I-MISC":8
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
        key= int(element.split(",")[0])
        value=" ".join(element.split(",")[1:])
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
            tmp_mapped=tmp_mapped.inner_pad(MAX_WORD_LEN,PADD_VALUE_CHAR)
            mapped.append(tmp_mapped)
        mapped = mapped.outer_pad(padded_list_length=MAX_WORD_LEN,pad_length=MAX_SENTENCE_LEN,pad_value=PADD_VALUE_CHAR)
        element["chars"] = mapped
        return [element]


class Label_parser(beam.DoFn):
    """Function to parse the labels"""
    def process(self,element):
        ### HERE FIX THE LABELS AND PARSE THEM CORRECTLY TO BREAK IT UP TO LIST
        ### THIS PART IS A MESS :) FIX IT UP NIKLAS
        labels = element["label"]
        text = element["text"]
        text = text.replace('"','').strip()
        text = ' '.join(text.split())
        element["text"] = text
        labels = labels.strip().split(" ")
        labels = [LABEL_MAPPING[label] for label in labels]
        labels=PadList(labels)
        element["labels"] =labels.inner_pad(MAX_SENTENCE_LEN, min(LABEL_MAPPING.values()))
        label_length = len(element["label"].split())
        sentence_length = len(element["text"].split())
        element["sentence_length"] = sentence_length if sentence_length<MAX_SENTENCE_LEN else MAX_SENTENCE_LEN
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

class ReadTextLablesPair(beam.PTransform):
    def expand(self,pcoll):
        return (pcoll |  "Merge the labels and the text for the train" >> beam.CoGroupByKey()
            | "Reshspae the train data " >> beam.ParDo(Reshape_data())
            | "Encode the train char data" >> beam.ParDo(Char_parser())
            | "Reshape the train labels " >> beam.ParDo(Label_parser())
            )

class DataStats(beam.PTransform):
    def expand(self,pcoll):
        return (pcoll |"Count the number of example" beam.combiners.CountCombineFn())

    # COMBINERS IS A GOOD THING!
    # STATEFUL IS A GOOD THING!
    # METRICS IS A GOOD THING!

    # The goal is to have:
    # 1) THe number of examples
    # 2) The number of positive and negative examples
    # 3) The lenght of the sentences, distribution
    # 4) The lenght of the words, distributions, grop by word lenght ish :) 
    # 5) The number of unique words before i encode them 




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
        file_train_data = "gs://named_entity_recognition/data/big_train/train_text.txt" 
        file_train_labels = "gs://named_entity_recognition/data/big_train/train_label.txt"
        
        file_test_data = "gs://named_entity_recognition/data/big_train/test_text.txt" 
        file_test_labels = "gs://named_entity_recognition/data/big_train/test_label.txt"


        train_text = (p | 'Read the train input data' >> beam.io.ReadFromText(file_train_data)
            | 'Set key value for train data' >> beam.ParDo(Set_Key_Value())
            )
        train_labels = (p | 'Read the labels for the train data' >> beam.io.ReadFromText(file_train_labels)  
            | 'Set key value for train labels' >> beam.ParDo(Set_Key_Value())
            )
        train_data = ({'text': train_text,'labels': train_labels}
            #| "Merge the labels and the text for the train" >> beam.CoGroupByKey()
            #| "Reshspae the train data " >> beam.ParDo(Reshape_data())
            #| "Encode the train char data" >> beam.ParDo(Char_parser())
            #| "Reshape the train labels " >> beam.ParDo(Label_parser())
             | ReadTextLablesPair()
            )

        test_text = (p | 'Read the test input data' >> beam.io.ReadFromText(file_test_data)
            | 'Set key value for test data' >> beam.ParDo(Set_Key_Value())
            )
        test_labels = (p | 'Read the labels for the test data' >> beam.io.ReadFromText(file_test_labels)  
            | 'Set key value for test labels' >> beam.ParDo(Set_Key_Value())
            )
        test_data = ({'text': test_text,'labels': test_labels}
            #| "Merge the labels and the text for the test" >> beam.CoGroupByKey()
            #| "Reshspae the test datahape " >> beam.ParDo(Reshape_data())
            #| "Encode the test char data" >> beam.ParDo(Char_parser())
            #| "Reshape the test labels " >> beam.ParDo(Label_parser())
             | ReadTextLablesPair()
            )

        train_dataset = (train_data, train_metadata)
        transformed_dataset, transform_fn = (train_dataset
                                              | 'AnalyzeAndTransform' >> beam_impl.AnalyzeAndTransformDataset(
                     preprocessing_fn))
        
        transformed_data, transformed_metadata = transformed_dataset

        transformed_data_coder = tft.coders.ExampleProtoCoder(
            transformed_metadata.schema)

        test_dataset = (test_data, train_metadata)
        transformed_test_data, _ = (
          (test_dataset, transform_fn)
          | 'Transform test data' >> beam_impl.TransformDataset())

        _ = (transformed_data
             | 'Encode train data to save it' >> beam.Map(transformed_data_coder.encode)
             | 'Write the train data to tfrecords' >> tfrecordio.WriteToTFRecord(os.path.join("gs://named_entity_recognition/beam/train","TRAIN"))
             )

        _ = (transformed_test_data
             | 'Encode test data to save it' >> beam.Map(transformed_data_coder.encode)
             | 'Write the test data to tfrecords' >> tfrecordio.WriteToTFRecord(os.path.join("gs://named_entity_recognition/beam/train","TEST"))
             )

        _ = (transform_fn
             | "WriteTransformFn" >>
             transform_fn_io.WriteTransformFn("gs://named_entity_recognition/beam/"))

        p.run().wait_until_finish()


if __name__=="__main__":
    main()