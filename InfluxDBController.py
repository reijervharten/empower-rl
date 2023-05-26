from influxdb import InfluxDBClient

class InfluxDBController(object):
    """Instantiate a connection to the InfluxDB."""

    def __init__(self):
        self.host = 'localhost'
        self.port = 8086
        self.user = 'root'
        self.password = 'root'
        self.dbname = 'empower'
        # self.pikey2 = 'B8:27:EB:95:92:76' #Pi_Key2 MAC 'E4:5F:01:12:AB:D1'
        # self.pikey1 = 'B8:27:EB:43:C0:54'  # Ubuntu Laptop 'A4:17:31:72:41:11' Pikey1 >>E4:5F:01:12:AB:38
        self.pi4 = 'DC:A6:32:FC:81:A9'  # Adaptor 'E8:4E:06:23:9B:EC My Mobile '98:F6:21:F6:6E:74
        self.query1 = 'SELECT mean("tx_bps") FROM "empower.apps.lvapbincounter.lvapbincounter" WHERE time >= now() - 30s and time <= now()'
        self.query2 = 'SELECT mean("tx_bps") FROM "empower.apps.lvapbincounter.lvapbincounter" WHERE ("sta"=$var_name) AND time >= now() - 30s and time <= now()'
        self.query3 = 'SELECT first("tx_bytes"), last("tx_bytes") ' + \
                            'FROM "empower.apps.wifislicestats.wifislicestats" ' + \
                            'WHERE time >= now() - 10s and time <= now() ' + \
                            'GROUP BY "slice_id"::tag '
        self.client = InfluxDBClient(self.host, self.port, self.user, self.password, self.dbname)

    #       print ("Connected to Influx DB Empower")

    def get_stats(self):
        # T2 = self.client.query(self.query2, bind_params={"var_name": self.pi4})
        # y2 = T2.raw['series'][0]['values'][0][1]

        results = []

        slice_throughputs = self.client.query(self.query3)
        for slice_stats in slice_throughputs.raw['series']:
            slice_id = slice_stats['tags']['slice_id']
            first = slice_stats['values'][0][1]
            last = slice_stats['values'][0][2]
            bandwidth = (last - first) / 30 / 1000000

            results.append((slice_id, bandwidth))
        
        results.sort(key=lambda x: x[0])

        return [8*throughput for [id, throughput] in results]