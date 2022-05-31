import pandas as pd
import csv

def test_answer():
    with open('resourse/file.csv', newline='') as csvfile:
        dfSource = pd.DataFrame(csv.reader(csvfile, delimiter=',', quotechar='|'))
        assert  len(dfSource) == 2026

def test_get_api_endpoint(self):
    r = self.app.get('/api/statistics')
    self.assertEqual(r.status_code, 200)