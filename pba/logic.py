# functions in this file act as adverbs for logic

def always(bool):

    if bool.__class__.__name__ == 'list':
        return False
    else:
        return bool

def sometimes(bool):

    if bool.__class__.__name__ == 'list':
        return True
    else:
        return bool
