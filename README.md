# Big-Data-Analytics
Repository to track the progress for the Big Data Analytics project. 
**Team:** Byte Me Analytics  
**Course:** Big Data Analytics @ WUT

## Project overview
This project implements a Lambda Architecture to analyze and predict the intensity of Wikipedia editing activity in real-time based on global news events (GDELT Project). The system correlates sentiment and volume of global news with edit spikes on specific Wikipedia pages to detect potential "edit wars."

### System architecture
* **Ingestion:** Apache NiFi (fetching GDELT & Wikipedia streams)
* **Message Broker:** Apache Kafka (Topic: `gdelt-events`)
* **Speed Layer:** Apache Spark Structured Streaming (Real-time inference)
* **Batch Layer:** Apache Spark MLlib (Model training on HDFS/Hive data)
* **Serving Layer:** Elasticsearch & Kibana


# Connecting to a virtual machine from the terminal
- `ssh big-data@100.98.48.77` and provide appropriate login and password

# Connecting to NiFi from the terminal
- `cd /opt/nifi`
- `nifi.sh start/stop/restart/status`

# Connecting to NiFi via https 
- `https://100.98.48.77:8443/nifi/` and provide appropriate login and password
