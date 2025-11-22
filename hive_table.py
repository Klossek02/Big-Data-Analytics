from pyhive import hive
from TCLIService.ttypes import TOperationState
import sys

HIVE_HOST = '100.98.48.77'
HIVE_PORT = 10000
HIVE_DB = 'default'

conn = hive.Connection(
    host=HIVE_HOST,
    port=HIVE_PORT,
    database=HIVE_DB
)
cursor = conn.cursor()

create_table_sql = """
CREATE EXTERNAL TABLE wikipedia_recentchanges_avro (
  id BIGINT,
  type STRING,
  namespace INT,
  title STRING,
  page_url STRING,
  comment STRING,
  event_timestamp BIGINT,
  username STRING,
  bot BOOLEAN,
  notify_url STRING,
  minor BOOLEAN,
  uri STRING,
  request_id STRING,
  meta_id STRING,
  domain STRING,
  stream STRING,
  topic STRING,
  partition_num INT,
  offset_used BIGINT,
  length_old INT,
  length_new INT,
  revision_old BIGINT,
  revision_new BIGINT,
  server_url STRING,
  server_name STRING,
  server_script_path STRING,
  wiki STRING,
  parsed_comment STRING
)
PARTITIONED BY (event_date STRING)
STORED AS AVRO
LOCATION '/big-data/hive/warehouse/wikipedia_recentchanges'
TBLPROPERTIES (
  'avro.schema.literal'='{
    "type": "record",
    "name": "wikipedia_recentchanges",
    "fields": [
      {"name": "id", "type": "long"},
      {"name": "type", "type": "string"},
      {"name": "namespace", "type": "int"},
      {"name": "title", "type": "string"},
      {"name": "page_url", "type": "string"},
      {"name": "comment", "type": "string"},
      {"name": "event_timestamp", "type": "long"},
      {"name": "username", "type": "string"},
      {"name": "bot", "type": "boolean"},
      {"name": "notify_url", "type": "string"},
      {"name": "minor", "type": "boolean"},
      {"name": "uri", "type": "string"},
      {"name": "request_id", "type": "string"},
      {"name": "meta_id", "type": "string"},
      {"name": "domain", "type": "string"},
      {"name": "stream", "type": "string"},
      {"name": "topic", "type": "string"},
      {"name": "partition_num", "type": "int"},
      {"name": "offset_used", "type": "long"},
      {"name": "length_old", "type": "int"},
      {"name": "length_new", "type": "int"},
      {"name": "revision_old", "type": "long"},
      {"name": "revision_new", "type": "long"},
      {"name": "server_url", "type": "string"},
      {"name": "server_name", "type": "string"},
      {"name": "server_script_path", "type": "string"},
      {"name": "wiki", "type": "string"},
      {"name": "parsed_comment", "type": "string"}
    ]
  }'
)
"""

try:
    cursor.execute(create_table_sql)
    print("Hive table 'wikipedia_recentchanges' created successfully.")
except Exception as e:
    print("Error creating table:", e, file=sys.stderr)
finally:
    cursor.close()
    conn.close()
