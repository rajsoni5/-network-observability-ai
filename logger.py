import psutil
import csv
import time
from datetime import datetime

def collect_network_data(filename="network_data.csv", interval=10):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'bytes_sent', 'bytes_recv', 'packets_sent', 'packets_recv'])

        while True:
            net = psutil.net_io_counters()
            writer.writerow([
                datetime.now().isoformat(),
                net.bytes_sent,
                net.bytes_recv,
                net.packets_sent,
                net.packets_recv
            ])
            file.flush()
            time.sleep(interval)

