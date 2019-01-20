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

def printy(element):
    print(element)

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cloud', type=str, help='y' )
    args = parser.parse_args(args=None) # Parse the arguments 
    
    if args.cloud=="y":
        pipeline_options = get_cloud_pipeline_options()
    else:
        pipeline_options = None
    p = beam.Pipeline(options=pipeline_options)
    filename_data = "gs://named_entity_recognition/data/raw/train.txt"
    raw_text = (p | 'Read input data' >> beam.io.ReadFromText(filename_data)
        )
    (raw_text | "Printy stage" >> beam.Map(printy))
    p.run().wait_until_finish()

if __name__=="__main__":
    main()