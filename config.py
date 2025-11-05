import datetime

# API credentials
username = '#'
password = '#'
base_url = "https://services.yesenergy.com/PS/rest/timeseries/"

start_date = datetime.datetime(2021,12,1)
end_date = datetime.datetime(2025,9,1)

params = {
    "startdate": start_date,
    "enddate": end_date,
    "agglevel": "hour"
}

PRICE_NODE_CONDOR = 20000003090      # condor
PRICE_NODE_SATICOY = 10016484939     # saticoy
PRICE_NODE_COSO = 20000002303        # coso
PRICE_NODE_SP15 = 20000004682        # sp15
