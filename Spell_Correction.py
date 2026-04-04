from sklearn.cluster import ward_tree


def correct_word(word):
    word = word.lower()

    # Step 1: Protect domain-specific words
    known_terms = ["dtr", "ai", "ml", "nlp", "etl"]

    if word in known_terms:
        return word

    # Step 2: Keep frequent words
    if word_freq.get(word, 0) > 3: # type: ignore
        return word

    # Step 3: Apply correction
    min_dist = float('inf')
    best_word = word

    for vocab_word in ward_tree:
        dist = edit_distance(word, vocab_word) # type: ignore

        if dist < min_dist or (dist == min_dist and word_freq[vocab_word] > word_freq.get(best_word, 0)): # type: ignore
            min_dist = dist
            best_word = vocab_word

    return best_word