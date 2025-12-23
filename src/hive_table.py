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
CREATE EXTERNAL TABLE wikipedia_edits (
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
STORED AS PARQUET
LOCATION '/big-data/hive/warehouse/wikipedia_edits'
TBLPROPERTIES (
  'parquet.compression'='SNAPPY'
)
"""

try:
    cursor.execute(create_table_sql)
    print("Hive table 'wikipedia_edits' created successfully.")
except Exception as e:
    print("Error creating table:", e, file=sys.stderr)
finally:
    cursor.close()
    conn.close()
