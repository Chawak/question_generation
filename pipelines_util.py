from difflib import SequenceMatcher
import re
from pythainlp.tokenize import word_tokenize
from pythainlp.tag import pos_tag
from nltk import word_tokenize as en_word_tokenize

def AD_BE_convert(context,answer):
    all_match=[(m.start(0), m.end(0)) for m in re.finditer("[0-9]+",answer)]
    for start,end in all_match[::-1]:
        if answer[start:end] not in context :
            if str(int(answer[start:end])+543) in context:
                answer = answer[:start]+str(int(answer[start:end])+543)+answer[end:]
            elif str(int(answer[start:end])-543) in context:
                answer = answer[:start]+str(int(answer[start:end])-543)+answer[end:]
    return answer


def add_unit_to_answer(context,answer,question):
    if context.count(answer)>1 and re.search("[0-9]+$",answer):
        if "กี่" in question:
            tokenize_question = word_tokenize(question)
            try :
                start_idx = tokenize_question.index("กี่")
                answer+=tokenize_question[start_idx+1]
                return answer
            except:
                pass

        if any(map(question.__contains__, ["เท่าไหร่" , "เท่าใด" ,"แค่ไหน" ,"เท่าไร"])) :
          tokenize_context = word_tokenize(context)
          pos_context = pos_tag(tokenize_context)

          try :
                start_idx = tokenize_context.index(answer)

                for tag_idx in range(start_idx+1,min(start_idx + 3,len(pos_context))) :
                  if pos_context[tag_idx][1]=="CMTR":
                    answer+=pos_context[tag_idx][0]
                    return answer
          except:
              pass

        for clause in ["how many","how much"]:
            if clause in question.lower():
                tokenize_question = en_word_tokenize(question)
                for i in range(2,len(tokenize_question)):
                    if tokenize_question[i-2].lower()=="how" and tokenize_question[i-1].lower()==clause.split()[1]:
                        tmp_answer = answer+" "+tokenize_question[i]
                        if context.count(tmp_answer)==1:
                            return tmp_answer
    return answer

def get_best_match_qa(query, corpus, step=4, flex=3, case_sensitive=False):
    """Return best matching substring of corpus.

    Parameters
    ----------
    query : str
    corpus : str
    step : int
        Step size of first match-value scan through corpus. Can be thought of
        as a sort of "scan resolution". Should not exceed length of query.
    flex : int
        Max. left/right substring position adjustment value. Should not
        exceed length of query / 2.

    Outputs
    -------
    output0 : str
        Best matching substring.
    output1 : float
        Match ratio of best matching substring. 1 is perfect match.
    """
    if query in corpus :
        return (query,1.0)
 
    def _match(a, b):
        """Compact alias for SequenceMatcher."""
        return SequenceMatcher(None, a, b).ratio()

    def scan_corpus(step):
        """Return list of match values from corpus-wide scan."""
        match_values = []

        m = 0
        while m + qlen - step <= len(corpus):
            match_values.append(_match(query, corpus[m : m+qlen]))
            m += step
        return match_values

    def index_max(v):
        """Return index of max value."""
        return max(range(len(v)), key=v.__getitem__)

    def adjust_left_right_positions():
        """Return left/right positions for best string match."""
        # bp_* is synonym for 'Best Position Left/Right' and are adjusted 
        # to optimize bmv_*
        p_l, bp_l = [pos] * 2
        p_r, bp_r = [pos + qlen] * 2

        # bmv_* are declared here in case they are untouched in optimization
        bmv_l = match_values[p_l // step]
        bmv_r = match_values[p_l // step]

        for f in range(flex):
            ll = _match(query, corpus[p_l - f: p_r])
            if ll > bmv_l:
                bmv_l = ll
                bp_l = p_l - f

            lr = _match(query, corpus[p_l + f: p_r])
            if lr > bmv_l:
                bmv_l = lr
                bp_l = p_l + f

            rl = _match(query, corpus[p_l: p_r - f])
            if rl > bmv_r:
                bmv_r = rl
                bp_r = p_r - f

            rr = _match(query, corpus[p_l: p_r + f])
            if rr > bmv_r:
                bmv_r = rr
                bp_r = p_r + f

        return bp_l, bp_r, _match(query, corpus[bp_l : bp_r])

    if not case_sensitive:
        query = query.lower()
        corpus = corpus.lower()

    qlen = len(query)

    # if flex >= qlen/2:
    #     print("Warning: flex exceeds length of query / 2. Setting to default.")
    #     flex = 3

    match_values = scan_corpus(step)
    pos = index_max(match_values) * step
    pos_left, pos_right, match_value = adjust_left_right_positions()

    return corpus[pos_left: pos_right].strip(), match_value
