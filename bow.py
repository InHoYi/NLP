import numpy as np

def wordToNumberDict(aListofWords: list) -> dict:
    result = {}
    for word in aListofWords:
        if word not in result:
            result[word] = len(result) + 1
    return result
