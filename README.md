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

## 2. Repository structure
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
│   ├── batch_gdelt_to_es.py # spark batch job to extract key entities from GDELT events and send them to elastic search
│   ├── batch_wiki_to_es.py  # spark batch job to extract minor vs major events and send them to elastic search
│   ├── wiki_stream.py       # Script for listening to the Wikipedia changes stream.
│   └── wikipedia_preprocessing.py # Spark batch job to preprocess raw Wikipedia data.
├── various/                 # Miscellaneous files.
└── README.md                # Detailed guide on description and environment config.
```

## 3. System requirements and prerequisites 
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
* **Elasticsearch and Kibana:** 8.6.x (at least)

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

### ElasticSearch and Kibana
Target directories: `etc/elasticsearch/` `etc/kibana/`
* `configs/elasticsearch.yml` -> properties for elastic search
* `configs/kibana.yml` -> properties for kibana


## 4. Service startup
Execute the following commands to start the Lambda Architecture components in the correct order.

### Connecting to a virtual machine from the terminal
- `ssh big-data@100.98.48.77` and provide appropriate login and password

### Connecting to NiFi via https 
- `https://100.98.48.77:8443/nifi/` and provide appropriate login and password

### Start Hadoop (HDFS and YARN)
```bash
# Start NameNode and DataNodes
start-dfs.sh

# Start ResourceManager and NodeManagers
start-yarn.sh

# Verify processes
jps
# Expected output should include: NameNode, DataNode, SecondaryNameNode, ResourceManager, NodeManager
```
### Start Hive 
```bash
# Run Hive Metastore as a background service
hive --service metastore &
```

### Start Apache Kafka (Zookeeper + Broker)
```bash
# Start Zookeeper
$KAFKA_HOME/bin/zookeeper-server-start.sh -daemon $KAFKA_HOME/config/zookeeper.properties

# Start Kafka Broker
$KAFKA_HOME/bin/kafka-server-start.sh -daemon $KAFKA_HOME/config/server.properties

# Create required topic (if not exist)
$KAFKA_HOME/bin/kafka-topics.sh --create --topic gdelt-events --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
$KAFKA_HOME/bin/kafka-topics.sh --create --topic wikipedia.edits --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1

# Launch consumer
/opt/kafka/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic gdelt-events --from-beginning
/opt/kafka/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic wikipedia.edits --from-beginning
```

### Start Apache NiFi
```bash
$NIFI_HOME/bin/nifi.sh start

# For stopping, restarting, and seeing status
$NIFI_HOME/bin/nifi.sh stop/restart/status
```

### Start Elasticsearch
```bash
sudo systemctl start elasticsearch
```

Or, if you want to start everything at once, launch the following script:
```
./start.sh

# For checking whether all the services have started
jps 
```


## Analytical module and streaming
### Python environment 

1. Launch the virtual environment:
```
source env/bin/activate  # for GDELT-related operations 
source wikienv/bin/activate # for Wiki-related operations 
```

2. Install required dependencies using pip:
```bash
pip install -r requirements.txt
# Key libraries: pyspark, elasticsearch<9.0.0, kafka-python, numpy
```

3. Launch one of the model scripts:
```
/opt/spark/bin/spark-submit --master local[*] train_model.py 2> /dev/null   # 2> /dev/null to make the output more eye-pleasing 
```

### Running the streaming pipeline 
1. Start the NiFi processors (GDELT_stream, Wiki_stream), Kafka server, and consumer with appropriate topic (either gdelt-events or wikipedia.edits), and launch the following Python file:
```python3 wiki_stream.py```

2. Submit the Spark streaming jobs:
```
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.2,org.elasticsearch:elasticsearch-spark-30_2.12:8.11.1 --driver-memory 512m --executor-memory 512m --num-executors 1 --executor-cores 1 --conf spark.sql.shuffle.partitions=2 streaming_wiki_to_es.py
```

```
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.2,org.elasticsearch:elasticsearch-spark-30_2.12:8.11.1 --driver-memory 512m --executor-memory 512m --num-executors 1 --executor-cores 1 --conf spark.sql.shuffle.partitions=2 streaming_gdelt_to_es.py
```

