import numpy as np

def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate

def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {} 
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word
    
def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        # print("word_id", word_id)
        for i in range(1, window_size + 1):
            # print("word_id", word_id)
            left_idx = idx - 1
            right_idx = idx + 1
            # print('right_idx', right_idx)

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                # print("left_word_id", left_word_id)
                co_matrix[word_id, left_word_id] += 1
                # print(co_matrix)
            
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
            

    return co_matrix

def cos_similarity(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(x**2)) + eps)
    ny = y / (np.sqrt(np.sum(y**2)) + eps)
    return np.dot(nx, ny)

def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):

    # クエリを取り消す
    if query not in word_to_id:
        print('%s is not found' % query)
        return 

    print('\n[query]' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # コサイン類似度の算出
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)

    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # コサイン類似度の結果から、その値を高い順に出力
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s : %s' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return 

def ppmi(C, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32)
    # print(C)
    N = np.sum(C)
    # print(N)
    S = np.sum(C, axis=0)
    # print(S)
    # print('C.shape[0]', C.shape[0], 'C.shape[1]', C.shape[1])
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j]*S[i]) + eps) # pmiの式
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total // 100) == 0:
                    print('%.1f%% done' % (100*cnt/total))

    return M
