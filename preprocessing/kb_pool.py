import time
import urllib, json
from SPARQLWrapper import SPARQLWrapper, JSON
import requests
from collections import defaultdict


# sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
url = "https://query.wikidata.org/sparql"

# From https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service/queries/examples#Cats
def single_query(term_list, subj):
  """relations with subject or object from the index """
  entity = '?s' if subj else '?o'
  # make a query with the term as subj OR obj:
  query = f"""
  SELECT ?s ?sLabel ?property ?propertyLabel ?o ?oLabel
  WHERE
  {{
    VALUES ?item {{ {term_list} }}
    {entity} rdfs:label | skos:altLabel ?item. # Look for both labels and aliases
    ?s ?p ?o.
    ?property wikibase:directClaim ?p .
    MINUS {{ ?property wikibase:propertyType wikibase:ExternalId . }} # Remode external identifiers from the result
    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
  }}
  ORDER BY ?s
  LIMIT 20
  """
  return apply_request(query)
  
def apply_request(query):
  r = requests.get(url, params = {'format': 'json', 'query': query})
  try:
    time.sleep(1)
    codes = dict()
    labels = dict()
    objects = dict()
    full_results_list = r.json()['results']['bindings']
    skip_if = ['defining formula', 'main subject', 'Stack Exchange tag', 'image', 'TeX string', 'in defining formula', 'described by source']
    skip_object = ['sourcing circumstances']
    html_chars = ['<', 'http', '\\', ':']
    for res in full_results_list:
      subj_label = res['sLabel']['value'].lower()
      obj_label = res['oLabel']['value'].lower()
      # defining some rules for the facts' content
      if res['propertyLabel']['value'] not in skip_if: 
        if subj_label not in skip_object and obj_label not in skip_object:
          if not any(n in res['oLabel']['value'] for n in html_chars):
            if len(obj_label.split()) < 4 and len(subj_label.split()) < 4:
        #and res['oLabel']['value'].islower() and res['sLabel']['value'].islower():
              # codes.append((res['s']['value'].split('/')[-1], res['property']['value'].split('/')[-1], res['o']['value'].split('/')[-1]))
              codes.setdefault(res['property']['value'].split('/')[-1], []).append((res['s']['value'].split('/')[-1],res['o']['value'].split('/')[-1]))
              labels.setdefault(res['property']['value'].split('/')[-1], []).append((subj_label, obj_label))
              # labels.setdefault(subj_label +'*****'+obj_label,[]).append('per: '+res['propertyLabel']['value'])
              # add object or subject to the set of possible entities:
              objects.update({obj_label:obj_label.split()})
              objects.update({subj_label:subj_label.split()})
    return codes, labels, objects
  except json.decoder.JSONDecodeError:
    print('json.decoder.JSONDecodeError: ')
    print(r)
    return [], dict(), dict()
  if r.status_code == 429:
    time.sleep(2)
    print('Stuck with code 429')
  if r.status_code == 430:
    print('Code 430')
    apply_request(query)


def rel_from_domain(relation, category):
  """
  "all facts holding this relation from certain domain

  relation: relation between searched facts
  category: domain name (math, CS, medicine, etc)
  rel_set: relations to look for with domain name
  """
  # query for instance_of foolowed by sublcass_of and domain category
  query = f"""
  SELECT DISTINCT ?s ?sLabel ?property ?propertyLabel ?o ?oLabel
  {{
    hint:Query hint:optimizer "None"
    VALUES ?property {{wdt:{relation}}}
    ?s wdt:P31* / wdt:P279* wd:{category} . # Find items in the domain
    ?s ?property ?o .
    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" . }}
  }}
  """
  return apply_request(query)
