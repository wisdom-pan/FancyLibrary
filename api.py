import requests

API_URL = "https://ai.gitee.com/api/endpoints/wisdompan/minicpm-v-2-6-7186/inference"
headers = {
	"Authorization": "Bearer eyJpc3MiOiJodHRwczovL2FpLmdpdGVlLmNvbSIsInN1YiI6IjQxMzgwIn0.d4K4aCY3FHsnNOcbbt9Hkd6valrsf2N3n3L8itrHz04QV0i_rEZFCR3GJOwhJ8nEQR8tiCEGkps_KxBDfjKJAg",
	"Content-Type": "application/json"
}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	print(response)
	return response.json()

output = query({
	"inputs": "Can you please let us know more details about your ",
})