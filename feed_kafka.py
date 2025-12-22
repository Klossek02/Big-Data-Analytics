
from kafka import KafkaProducer
import json
import time
import random

producer = KafkaProducer(value_serializer=lambda x: json.dumps(x).encode('utf-8'))

print("Sending data to Kafka...")
for i in range(10):
    data = {
        "SOURCEURL": f"http://test-news-{i}.com",
        "AvgTone": random.uniform(-10, 10),
        "PositiveScore": random.uniform(0, 5),
        "NegativeScore": random.uniform(0, 5),
        "SQLDATE": "20251220"
    }
    producer.send('gdelt-events', value=data)
    time.sleep(0.5)

producer.flush()
print("Sent 10 records.")