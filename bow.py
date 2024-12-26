import numpy as np

def deleteDuplicates(aListofWords: list) -> list:
    result = []
    for word in aListofWords:
        if word not in result:
            result.append(word)
    
    return result


def deleteDuplicatesbySet(aListofWords: list) -> list:
    result = set(aListofWords)

    return result


def wordToNumberDict(aListofWords: list) -> dict:
    result = {}
    for word in aListofWords:
        if word not in result:
            result[word] = len(result) + 1

    return result


def numberToWordDict(aListofWords: list) -> dict:
    result = {}
    for word in aListofWords:
        if word not in result.values():
            result[len(result) + 1] = word

    return result


def oneHotEncoding(aListofWords: list) -> dict:
    result = {}
    uniqueSet = deleteDuplicatesbySet(aListofWords)
    values = np.eye(len(uniqueSet))
    result = dict(zip(uniqueSet, values))
    
    return result