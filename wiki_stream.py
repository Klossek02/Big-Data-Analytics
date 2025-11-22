#!/home/big-data/wikienv/bin/python3
import json
import time
import requests
from pywikibot.comms.eventstreams import EventStreams


NIFI_URL = "http://100.98.48.77:5000/contentListener"
REQUEST_TIMEOUT = 10

STREAMS = ['recentchange']
WIKI_SERVER = 'en.wikipedia.org'
FILTER_TYPE = 'edit'
RECONNECT_TIMEOUT = 60
stream = EventStreams(streams=STREAMS)
stream.register_filter(server_name=WIKI_SERVER, type=FILTER_TYPE)

while True:
    try:
        for change in stream:
            if not change:
                continue

            # skip canary events
            if change.get("meta", {}).get("domain") == "canary":
                continue

            # print for debug
            print(f"{change['type']} on page '{change['title']}' by '{change['user']}' at {change['meta']['dt']}", flush=True)

            # send to NiFi
            try:
                resp = requests.post(
                    NIFI_URL,
                    json=change,
                    timeout=REQUEST_TIMEOUT
                )
                if resp.status_code != 200:
                    print(f"Warning: NiFi returned {resp.status_code}", flush=True)
            except requests.exceptions.RequestException as e:
                print(f"HTTP error: {e}, retrying in 5s...", flush=True)
                time.sleep(5)
                continue

    except Exception as e:
        print(f"Stream error: {e}, reconnecting in {RECONNECT_TIMEOUT}s...", flush=True)
        time.sleep(RECONNECT_TIMEOUT)
