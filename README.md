# Big Data Analytics project
Repository to track the progress for the Big Data Analytics project. 

**Team:** Byte Me Analytics  
**Course:** Big Data Analytics @ WUT

## 1. Project overview
This project implements a Lambda Architecture to analyze and predict the intensity of Wikipedia editing activity in real-time based on global news events (GDELT Project). The system correlates sentiment and volume of global news with edit spikes on specific Wikipedia pages to detect potential "edit wars".

### System architecture
* **Ingestion:** Apache NiFi (fetching GDELT and Wikipedia streams)
* **Message broker:** Apache Kafka (topics: `gdelt-events`, `wikipedia.edits`)
* **Speed layer:** Apache Spark structured streaming (real-time inference)
* **Batch layer:** Apache Spark MLlib (model training on HDFS/Hive data)
* **Serving layer:** Elasticsearch and Kibana

## Repository structure
```
├── EDA/                     # Folder for Exploratory Data Analysis (EDA) .ipynb files, made for Milestone 3.
├── configs/                 # Essential cluster configuration files for Hadoop, Spark, Kafka, NiFi.
├── nifi_flows/              # Apache NiFi flow definitions (.xml) for GDELT and Wikipedia pipelines.
├── src/                     # Source code for Spark jobs and Python scripts.
│   ├── feed_kafka.py        # Script generating mock data that simulates GDELT news stream.
│   ├── gdelt_preprocess.py  # Spark batch job to preprocess raw GDELT data on HDFS.
│   ├── gdelt_preprocessing.py # Local version of GDELT data processing for testing.
│   ├── hive_table.py        # Script creating external Hive tables (SQL).
│   ├── inspect_model.py     # Script for inspecting trained model weights.
│   ├── model_training.py    # Main script for training ML models with augmentation.
│   ├── model_training_debug.py # ML training script with verbose debugging.
│   ├── model_training_no_augmentation.py # ML training script using only real data.
│   ├── streaming_to_es.py   # Real-time streaming job with Elasticsearch sink.
│   ├── wiki_stream.py       # Script for listening to Wikipedia changes stream.
│   └── wikipedia_preprocessing.py # Spark batch job to preprocess raw Wikipedia data.
├── various/                 # Miscellaneous files.
└── README.md                # Detailed guide on description and environment config.
```

## 2. System requirements and prerequisites 
* **OS:** Linux (Ubuntu 22.04 LTS recommended)
* **Java:** OpenJDK 8 or 11
* **Python:** 3.8+
* **Network:** Tailscale 

### Component versions 
The system relies on the following software versions:
* **Apache Hadoop:** 3.3.6
* **Apache Spark:** 3.5.0
* **Apache Kafka:** 3.5.0 (Scala 2.12)
* **Apache NiFi:** 1.23.2
* **Elasticsearch and Kibana:** 8.13.x

### Configuration management
As mentioned, critical configuration files are versioned in the `configs/` directory of this repository. To provision the environment, symlink or copy these files to their respective service configuration directories.

### Hadoop and YARN
Target directory: `$HADOOP_HOME/etc/hadoop/`
* `configs/core-site.xml` -> system core settings (HDFS address)
* `configs/hdfs-site.xml`  -> HDFS replication and path settings
* `configs/hive-site.xml`  -> Hive replication and path settings
* `configs/yarn-site.xml` -> ResourceManager and NodeManager config
* `configs/mapred-site.xml` -> MapReduce framework settings

### Apache Spark
Target directory: `$SPARK_HOME/conf/`
* `configs/spark-defaults.conf` -> default execution settings (executor memory, driver settings, etc.)

### Apache NiFi
Target directory: `$NIFI_HOME/conf/`
* `configs/nifi.properties` -> fundamental NiFi properties (ports, repository paths, etc.)


## 3. How to run 

## Essential commands 
# Connecting to a virtual machine from the terminal
- `ssh big-data@100.98.48.77` and provide appropriate login and password

# Connecting to NiFi from the terminal
- `cd /opt/nifi`
- `nifi.sh start/stop/restart/status`

# Connecting to NiFi via https 
- `https://100.98.48.77:8443/nifi/` and provide appropriate login and password
