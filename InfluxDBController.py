from datetime import datetime
from influxdb import InfluxDBClient

class InfluxDBController(object):
    """Instantiate a connection to the InfluxDB."""

    def __init__(self):
        self.host = 'localhost'
        self.port = 8086
        self.user = 'root'
        self.password = 'root'
        self.dbname = 'empower'
        self.pi4 = 'DC:A6:32:FC:81:A9'  # Adaptor 'E8:4E:06:23:9B:EC My Mobile '98:F6:21:F6:6E:74
        self.query_now = 'SELECT last("tx_bytes") ' + \
                            'FROM "empower.apps.wifislicestats.wifislicestats" ' + \
                            'WHERE time >= now() - 4s and time <= now() ' + \
                            'GROUP BY "slice_id"::tag '
        self.query_prev = 'SELECT last("tx_bytes") ' + \
                            'FROM "empower.apps.wifislicestats.wifislicestats" ' + \
                            'WHERE time <= now() - 4s ' + \
                            'GROUP BY "slice_id"::tag '
        self.client = InfluxDBClient(self.host, self.port, self.user, self.password, self.dbname)

    #       print ("Connected to Influx DB Empower")

    def get_stats(self):
        results = []

        slice_bytes_now = self.client.query(self.query_now)
        slice_bytes_prev = self.client.query(self.query_prev)
        for slice_stats in slice_bytes_now.raw['series']:
            slice_id = slice_stats['tags']['slice_id']

            bytes_now = slice_stats['values'][0][1]
            time_now = datetime.strptime(slice_stats['values'][0][0], '%Y-%m-%dT%H:%M:%S.%fZ')
            bytes_prev = None
            for prev_stats in slice_bytes_prev.raw['series']:
                slice_id_prev = prev_stats['tags']['slice_id']
                if (slice_id == slice_id_prev):
                    bytes_prev = prev_stats['values'][0][1]
                    time_prev = datetime.strptime(prev_stats['values'][0][0], '%Y-%m-%dT%H:%M:%S.%fZ')
                    break
            
            if bytes_prev != None and bytes_now >= bytes_prev:
                d_time = (time_now - time_prev).total_seconds()
                bandwidth = 8*(bytes_now - bytes_prev) / d_time / 1000000

                results.append((slice_id, bandwidth))
        
        results.sort(key=lambda x: x[0])

        return results