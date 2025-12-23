# Big Data Analytics project
Repository to track the progress for the Big Data Analytics project. 

**Team:** Byte Me Analytics  
**Course:** Big Data Analytics @ WUT

## Project overview
This project implements a Lambda Architecture to analyze and predict the intensity of Wikipedia editing activity in real-time based on global news events (GDELT Project). The system correlates sentiment and volume of global news with edit spikes on specific Wikipedia pages to detect potential "edit wars".

### System architecture
* **Ingestion:** Apache NiFi (fetching GDELT and Wikipedia streams)
* **Message broker:** Apache Kafka (topics: `gdelt-events`, `wikipedia.edits`)
* **Speed layer:** Apache Spark structured streaming (real-time inference)
* **Batch layer:** Apache Spark MLlib (model training on HDFS/Hive data)
* **Serving layer:** Elasticsearch and Kibana

## Repository structure
```
* **`/EDA`** - folder for Exploratory Data Analysis (EDA) .ipynb files, made for Milestone 3.  
* **`/configs`** - essential cluster configuration files for Hadoop, Spark, Kafka, NiFi.
* **`/nifi_flows`** - Apache NiFi flow definitions (.xml) for GDELT and Wikipedia pipelines.
* **`/src`** - source code for Spark jobs and Python scripts.
    * `feed_kafka.py` - script generating mock data that pretends to be a news stream from the GDELT system for test purposes.
    * `gdelt_preprocess.py` - script for Spark batch job to preprocess raw GDELT data previously downloaded to HDFS/Hive.
    * `gdelt_preprocessing.py`- script for raw GDELT data processing, but tested locally.
    * `hive_table.py` - script which connects to the Hive server and executes the `SQL CREATE EXTERNAL TABLE` command.
    * `inspect_model.py` - script for looking inside a trained model.
    * `model_training.py` - script for training our machine learning (ML) models. 
    * `model_training_debug.py` - script for training our ML models with the addition of more sophisticated debug commands. 
    * `model_training_no_augmentation.py` - script for training our ML models without augmentation logic provided. 
    * `streaming_to_es.py` - script for a real-time streaming job with an Elasticsearch sink.
    * `wiki_stream.py` - script for data "listening" from Wikipedia. 
    * `wikipedia_preprocessing.py` - script for Spark batch job to preprocess raw Wikipedia data previously downloaded to HDFS/Hive.
* **`/various`** - for miscallenous files. 
* **`README.md` - detailed guide on the description and configuring the environment.
```

## Essential commands 
# Connecting to a virtual machine from the terminal
- `ssh big-data@100.98.48.77` and provide appropriate login and password

# Connecting to NiFi from the terminal
- `cd /opt/nifi`
- `nifi.sh start/stop/restart/status`

# Connecting to NiFi via https 
- `https://100.98.48.77:8443/nifi/` and provide appropriate login and password
