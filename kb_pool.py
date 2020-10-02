from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

# From https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service/queries/examples#Cats
def single_query(word):
  query = f"""
  SELECT ?s ?sLabel ?property ?propertyLabel ?o ?oLabel
  WHERE
  {{
    ?s rdfs:label "{word}"@en .
    ?s ?p ?o .
    ?property wikibase:directClaim ?p .
    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
  }}
  """
  sparql.setQuery(query)
  sparql.setReturnFormat(JSON)
  results = sparql.query().convert()

  print(results['results']['bindings'])

single_query('optimization')