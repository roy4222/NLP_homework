1.no [('geese', <Noun_number.SINGULAR: 1>), ('road', <Noun_number.SINGULAR: 1>)]
2.yes [('geese', <Noun_number.PLURAL: 2>), ('road', <Noun_number.SINGULAR: 1>)]

The morph method gives incorrect results because `morph.get("Number")` returns a list (e.g. `['Plur']`), not a string. Comparing it with the string `"Sing"` is always False, so all nouns are classified as SINGULAR. Additionally, the logic in the lecture code is inverted — it assigns PLURAL when it detects "Sing".

The lemma method works correctly because spaCy knows the lemma of "geese" is "goose". Since goose ≠ geese, it is correctly identified as PLURAL.
