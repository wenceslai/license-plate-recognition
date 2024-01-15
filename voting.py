import utils

def majority_voting(plates):
    dict = {}
    keys_to_delete =[]

    for plate in plates:
        if plate in dict:
            dict[plate] = dict[plate] + 1
        else:
            dict[plate] = 1

    for key1 in dict:
        for key2 in dict:
            if key1 != key2:
                if utils.similar_strings(key1, key2):
                    min_key = ""
                    if dict[key2] > dict[key1]:
                        min_key = key1
                    else:
                        min_key = key2
                    if min_key not in keys_to_delete:
                        keys_to_delete.append(min_key)

    for key in keys_to_delete:
        del dict[key]

    return list(dict.keys())
