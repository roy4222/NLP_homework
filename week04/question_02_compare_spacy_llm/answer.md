1. yes
   SpaCy's large model (`en_core_web_lg`) correctly identifies "geese" as plural (lemma=goose, morph=Number=Plur, tag=NNS).
   Lemma method output: [('geese', <Noun_number.PLURAL: 2>), ('road', <Noun_number.SINGULAR: 1>)]

2. yes, Claude
   LLM can easily determine that "geese" is the plural form of "goose" — this is basic linguistic knowledge for language models.

3. SpaCy is faster
   SpaCy processes locally in about 0.01 seconds, while LLM requires an API call that typically takes 0.5 seconds or more. The difference is significant due to network round-trip overhead.
