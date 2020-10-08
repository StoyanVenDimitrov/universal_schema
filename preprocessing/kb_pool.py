import time
import urllib, json
from SPARQLWrapper import SPARQLWrapper, JSON
import requests

# sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
url = "https://query.wikidata.org/sparql"

# From https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service/queries/examples#Cats
def single_query(term_list):
  query = f"""
  SELECT ?s ?sLabel ?property ?propertyLabel ?o ?oLabel
  WHERE
  {{
    VALUES ?item {{ {term_list} }}
    ?s rdfs:label | skos:altLabel ?item. # Look for both labels and aliases
    ?s ?p ?o .
    ?property wikibase:directClaim ?p .
    MINUS {{ ?property wikibase:propertyType wikibase:ExternalId . }} # Remode external identifiers from the result
    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
  }}
  ORDER BY ?s
  """
  r = requests.get(url, params = {'format': 'json', 'query': query})
  try:
    time.sleep(1)
    codes = []
    labels = []
    full_results_list = r.json()['results']['bindings']
    skip_if = ['defining formula', 'Stack Exchange tag', 'image', 'TeX string', 'in defining formula', 'described by source']
    for res in full_results_list:
      if res['propertyLabel']['value'] not in skip_if:
        codes.append((res['s']['value'], res['property']['value'], res['o']['value']))
        labels.append((res['sLabel']['value'], res['propertyLabel']['value'], res['oLabel']['value']))
    print(term_list)
    return codes, labels
  except json.decoder.JSONDecodeError:
    print(term_list)
  if r.status_code == 429:
    time.sleep(2)
    print('Stuck with code 429')
  if r.status_code == 430:
    print('Code 430')
    single_query(term_list)

  # try:
  #   #time.sleep(1)
  #   sparql.setQuery(query)
  #   sparql.setReturnFormat(JSON)
  #   results = sparql.query().convert()

  #   return results['results']['bindings'][0]
  # except ValueError as error:
  #   print(results['header'])
  #   single_query(word)


  
