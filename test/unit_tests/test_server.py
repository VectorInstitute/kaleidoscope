import requests

def test_ping():
    "Verify the existance of Lingua server"
    server_url= 'http://llm.cluster.local:3001/'
    assert requests.get(server_url).ok