import dask.dataframe as dd


def main():
    df = dd.read_csv("gs://named_entity_recognition/beam/tftransform_tmp/vocab_compute_and_apply_vocabulary_vocabulary",header=None)
    df = df.reset_index()
    df.columns = ["index","Word"]
    df.to_csv(".",sep="\t",index=False)
if __name__=="__main__":
    main()