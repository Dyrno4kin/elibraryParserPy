import pandas as pd
import csv
import unittest
import server as tested_app
import json

class FlaskAppTests(unittest.TestCase):

    def setUp(self):
        tested_app.app.config['TESTING'] = True
        self.app = tested_app.app.test_client()

    # def test_answer():
    #     with open('resourse/file.csv', newline='') as csvfile:
    #         dfSource = pd.DataFrame(csv.reader(csvfile, delimiter=',', quotechar='|'))
    #         assert len(dfSource) == 2026

    def test_get_api_endpoint(self):
        r = self.app.get('/api/statistics')
        self.assertEqual(r.status_code, 200)



if __name__ == '__main__':
    unittest.main()
